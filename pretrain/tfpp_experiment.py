# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import math
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import lightning as L
import numpy as np
import torch
from lightning.fabric.loggers import CSVLogger
from pytorch_lightning.loggers import WandbLogger
from lightning.fabric.strategies import DDPStrategy
from lightning.fabric.utilities import ThroughputMonitor, measure_flops
from torch.utils.data import DataLoader, IterableDataset

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from lit_gpt import Config
from lit_gpt.model_exp import GPT, Block
from lit_gpt.utils import chunked_cross_entropy, estimate_flops, get_default_supported_precision, num_parameters

model_name = "tfpp_elementwise"
name = "tfpp_elementwise"
config_file = "configs/tfpp_experiment.json"
out_dir = Path(f"/scratch/oymak_root/oymak0/milii/owt_mistral/{name}")
#data_dir = Path("/nfs/turbo/coe-sodalab/shared_data/owt_mistral")
# data_dir = Path("/scratch/oymak_root/oymak0/milii/datasets/openwebtext")
data_dir = Path("/tmpssd/milii/datasets/openwebtext")
save_interval = 1000
eval_interval = 1000
eval_iters = 200
log_interval = 10

# Hyperparameters
learning_rate = 6e-4
batch_size = 480
micro_batch_size = 12
gradient_accumulation_steps = 0
max_iters = 600000   # num_epochs * (epoch_size // micro_batch_size) // devices
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
decay_lr = True
warmup_iters = 2000
lr_decay_iters = max_iters
min_lr = 6e-5

hparams = {k: v for k, v in locals().items() if isinstance(v, (int, float, str)) and not k.startswith("_")}
# logger = CSVLogger("out", name, flush_logs_every_n_steps=log_interval)
logger = WandbLogger(name, save_dir="out", project="gateRes" )


def setup(devices: int = 1, precision: Optional[str] = None, resume: Union[bool, Path] = False) -> None:
    precision = precision or get_default_supported_precision(training=True)

    if devices > 1:
        parallel_devices = [torch.device(f"cuda:{i}") for i in range(devices)]
        strategy = DDPStrategy( parallel_devices=parallel_devices, precision=precision)
    else:
        strategy = "auto"

    fabric = L.Fabric(devices=devices, strategy=strategy, precision=precision, loggers=logger)

    if hparams["gradient_accumulation_steps"] ==0:
        assert batch_size % micro_batch_size == 0
        gradient_accumulation_steps = batch_size // micro_batch_size
        assert gradient_accumulation_steps % devices == 0
        gradient_accumulation_steps = gradient_accumulation_steps // devices
        assert gradient_accumulation_steps > 0
        hparams["gradient_accumulation_steps"] = gradient_accumulation_steps
    hparams["devices"] = devices
    fabric.print(hparams)
    fabric.launch(main, resume=resume)


def main(fabric: L.Fabric, resume: Union[bool, Path]) -> None:
    if fabric.global_rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    fabric.seed_everything(1337, workers=True)  # same seed for every process to init model (FSDP)

    config = Config.from_json(config_file)
    fabric.print(f"Loading model with {config.__dict__}")
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=(fabric.world_size > 1)):
        model = GPT(config)
    model.apply(model._init_weights)

    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.")
    fabric.print(f"Total parameters {num_parameters(model):,}")

    model = fabric.setup(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(beta1, beta2), foreach=False
    )
    optimizer = fabric.setup_optimizers(optimizer)

    train_data, val_data = load_datasets(data_dir, max_seq_length=model.max_seq_length)
    train_dataloader = DataLoader(train_data, batch_size=micro_batch_size, num_workers=2)
    val_dataloader = DataLoader(val_data, batch_size=micro_batch_size, num_workers=2)
    train_dataloader, val_dataloader = fabric.setup_dataloaders(train_dataloader, val_dataloader)

    state = {"model": model, "optimizer": optimizer, "hparams": hparams, "iter_num": 0, "step_count": 0}

    if resume is True:
        resume = max(out_dir.glob("*.pth"), key=lambda p: int(p.name.split("-")[1]))
    if resume:
        fabric.print(f"Resuming training from {resume}")
        fabric.load(resume, state)

    train_time = time.perf_counter()
    train(fabric, state, train_dataloader, val_dataloader)
    fabric.print(f"Training time: {(time.perf_counter()-train_time):.2f}s")
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")


def train(fabric: L.Fabric, state: dict, train_dataloader: DataLoader, val_dataloader: DataLoader) -> None:
    model = state["model"]
    optimizer = state["optimizer"]

    validate(fabric, model, val_dataloader, max_iters=2)  # sanity check

    with torch.device("meta"):
        meta_model = GPT(model.config)
        # "estimated" is not as precise as "measured". Estimated is optimistic but widely used in the wild.
        # When comparing MFU or FLOP numbers with other projects that use estimated FLOPs,
        # consider passing `flops_per_batch=estimated_flops` instead
        estimated_flops = estimate_flops(meta_model, training=True) * micro_batch_size
        fabric.print(f"Estimated TFLOPs: {estimated_flops * fabric.world_size / 1e12:.2f}")
        x = torch.randint(0, 1, (micro_batch_size, model.max_seq_length))
        forward_fn = lambda: meta_model(x)
        loss_fn = lambda y: chunked_cross_entropy(y, x, chunk_size=0)
        measured_flops = measure_flops(meta_model, forward_fn, loss_fn)
        fabric.print(f"Measured TFLOPs: {measured_flops * fabric.world_size / 1e12:.2f}")
        del meta_model, x

    throughput = ThroughputMonitor(fabric, window_size=50)
    total_t0 = time.perf_counter()

    train_iter = iter(train_dataloader)

    for state["iter_num"] in range(state["iter_num"], max_iters):
        iter_num = state["iter_num"]
        lr = get_lr(iter_num, warmup_iters, lr_decay_iters) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if iter_num % eval_interval == 0:
            t0 = time.perf_counter()
            val_loss = validate(fabric, model, val_dataloader, max_iters=eval_iters)
            print("val_loss", val_loss)
            train_loss = validate(fabric, model, train_dataloader, max_iters=eval_iters)
            print("train_loss", train_loss)
            t1 = time.perf_counter() - t0
            fabric.log_dict(metrics = {"eval/val_loss": val_loss.item(),
                                       "eval/time": t1 * 1000,
                                       "eval/train_loss": train_loss.item(),
                                       "eval/lr": lr,
                                       "step": iter_num,
                                       "eval/valtime": t1 * 1000}, step=iter_num//log_interval)
            fabric.print(f"step {iter_num}: val loss {val_loss.item():.4f}, val time: {t1 * 1000:.2f}ms")
            fabric.barrier()
        if iter_num % save_interval == 0:
            checkpoint_path = out_dir / f"iter-{iter_num:06d}-ckpt.pth"
            fabric.print(f"Saving checkpoint to {str(checkpoint_path)!r}")
            fabric.save(checkpoint_path, state)

        iter_t0 = time.perf_counter()

        gradient_accumulation_steps = hparams["gradient_accumulation_steps"]

        for micro_step in range(gradient_accumulation_steps):
            is_accumulating = micro_step == gradient_accumulation_steps - 1
            with fabric.no_backward_sync(model, enabled=is_accumulating):
                input_ids, targets = next(train_iter)
                logits = model(input_ids)
                loss = chunked_cross_entropy(logits, targets, chunk_size=0)
                fabric.backward(loss / gradient_accumulation_steps)
        fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
        optimizer.step()
        optimizer.zero_grad()

        if iter_num % log_interval == 0:
            loss_item = loss.item()  # expensive device-to-host synchronization
            t1 = time.perf_counter()
            throughput.update(
                time=t1 - total_t0,
                batches=iter_num,
                samples=iter_num * batch_size,
                lengths=iter_num * batch_size * model.max_seq_length,
                flops=measured_flops * log_interval * gradient_accumulation_steps,
            )
            t_used = t1 - total_t0
            est_time = t_used / (iter_num + 1) * max_iters - t_used
            #throughput.compute_and_log(step=iter_num//log_interval)
            fabric.log_dict(metrics = {"running/iter": iter_num,
                                       "running/loss": loss_item,
                                       "running/lr": lr,
                                        "running/remaining_time": est_time / 60. / 60.,
                                       "running/itertime": (t1 - iter_t0) * 1000,
                                       "step": iter_num}, step=iter_num//log_interval)
            fabric.print(
                f"iter {iter_num}: loss {loss_item:.4f}, iter time:"
                f" {(t1 - iter_t0) * 1000:.2f}ms, est. time remaining: {est_time / 60. / 60.:.2f}h"
            )


# FSDP has issues with `inference_mode`
@torch.no_grad()
def validate(fabric: L.Fabric, model: torch.nn.Module, val_dataloader: DataLoader, max_iters: int) -> torch.Tensor:
    fabric.print("Validating ...")
    model.eval()
    val_iter = iter(val_dataloader)

    losses = torch.zeros(max_iters, device=fabric.device)
    for k in range(max_iters):
        input_ids, targets = next(val_iter)
        logits = model(input_ids)
        losses[k] = chunked_cross_entropy(logits, targets, chunk_size=0)
    out = losses.mean()

    model.train()
    return out


def load_datasets(data_dir: Path, max_seq_length: int) -> Tuple["Dataset", "Dataset"]:
    train_data = Dataset(data_dir / "train.bin", max_seq_length)
    val_data = Dataset(data_dir / "val.bin", max_seq_length)
    return train_data, val_data


class Dataset(IterableDataset):
    def __init__(self, data_file: Path, max_seq_length: int):
        super().__init__()
        self.data_file = data_file
        self.max_seq_length = max_seq_length

    def __iter__(self):
        data = np.memmap(self.data_file, dtype=np.uint16, mode="r")
        while True:
            i = torch.randint(len(data) - self.max_seq_length, (1,)).item()
            x = torch.from_numpy((data[i : i + self.max_seq_length]).astype(np.int64))
            y = torch.from_numpy((data[i + 1 : i + 1 + self.max_seq_length]).astype(np.int64))
            yield x, y


# learning rate decay scheduler (cosine with linear warmup)
def get_lr(it: int, warmup_iters: int, max_iters: int) -> float:
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > max_iters, return min learning rate
    if it > max_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    from jsonargparse import CLI

    CLI(setup)
