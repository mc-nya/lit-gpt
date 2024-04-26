# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import sys
import time
from pathlib import Path
from typing import Literal, Optional

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

# from generate.base import generate
# from lit_gpt import GPT, Config, Tokenizer
from lit_gpt import Tokenizer, Config
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, load_checkpoint
from scripts.prepare_alpaca import generate_prompt


def main(
    prompt: str = "What food do llamas eat?",
    input: str = "",
    finetuned_path: Path = Path(""),
    checkpoint_dir: Path = Path(""),
    config_filepath: Path = Path(""),
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    max_new_tokens: int = 100,
    top_k: Optional[int] = 200,
    temperature: float = 0.8,
    precision: Optional[str] = None,
    model_type: str = "standard"
) -> None:
    """Generates a response based on a given instruction and an optional input.
    This script will only work with checkpoints from the instruction-tuned GPT model.
    See `finetune/full.py`.

    Args:
        prompt: The prompt/instruction (Alpaca style).
        input: Optional input (Alpaca style).
        finetuned_path: Path to the checkpoint with trained weights, which are the output of
            `finetune/full.py`.
        checkpoint_dir: The path to the checkpoint folder with pretrained GPT weights.
        quantize: Whether to quantize the model and using which method:
            - bnb.nf4, bnb.nf4-dq, bnb.fp4, bnb.fp4-dq: 4-bit quantization from bitsandbytes
            - bnb.int8: 8-bit quantization from bitsandbytes
            for more details, see https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/quantize.md
        max_new_tokens: The number of generation steps to take.
        top_k: The number of top most probable tokens to consider in the sampling process.
        temperature: A value controlling the randomness of the sampling process. Higher values result in more random
            samples.
        precision: Indicates the Fabric precision setting to use.
    """
    precision = precision or get_default_supported_precision(training=False)

    is_pe = not "nope" in model_type
    is_conv = "conv" in model_type
    print("is_pe", is_pe, "is_conv", is_conv)
    if is_pe and is_conv:
        print("Using model defined in model_conv_pe.py")
        from lit_gpt.model_conv_pe import GPT, Block
    elif is_pe and not is_conv:
        print("Using model defined in model.py")
        from lit_gpt.model import GPT, Block
    elif not is_pe and is_conv:
        print("Using model defined in model_conv_nope.py")
        from lit_gpt.model_conv_nope import GPT, Block
    elif not is_pe and not is_conv:
        print("Using model defined in model_nope.py")
        from lit_gpt.model_nope import GPT, Block

    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError("Quantization and mixed precision is not supported.")
        dtype = {"16-true": torch.float16, "bf16-true": torch.bfloat16, "32-true": torch.float32}[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)
    fabric.launch()

    check_valid_checkpoint_dir(checkpoint_dir)

    # config = Config.from_json(checkpoint_dir / "lit_config.json")
    config = Config.from_json(config_filepath)

    checkpoint_path = finetuned_path

    tokenizer = Tokenizer(checkpoint_dir)
    sample = {"instruction": prompt, "input": input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=fabric.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
    t0 = time.perf_counter()
    with fabric.init_module(empty_init=True):
        model = GPT(config)
    fabric.print(f"Time to instantiate model: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)
    with fabric.init_tensor():
        # set the max_seq_length to limit the memory usage to what we need
        model.max_seq_length = max_returned_tokens
        # enable the kv cache
        # model.set_kv_cache(batch_size=1)
    model.eval()

    model = fabric.setup(model)

    t0 = time.perf_counter()
    load_checkpoint(fabric, model, checkpoint_path)
    fabric.print(f"Time to load the model weights: {time.perf_counter() - t0:.02f} seconds.", file=sys.stderr)

    L.seed_everything(1234)
    t0 = time.perf_counter()
    y = generate(model, encoded, max_returned_tokens, temperature=temperature, top_k=top_k, eos_id=tokenizer.eos_id)
    t = time.perf_counter() - t0

    output = tokenizer.decode(y)
    #output = output.split("### Response:")[1].strip()
    fabric.print(output)

    tokens_generated = y.size(0) - prompt_length
    fabric.print(f"\n\nTime for inference: {t:.02f} sec total, {tokens_generated / t:.02f} tokens/sec", file=sys.stderr)
    if fabric.device.type == "cuda":
        fabric.print(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB", file=sys.stderr)

from generate.base import sample
def next_token(model, x: torch.Tensor, **kwargs) -> torch.Tensor:
    logits = model(x)
    next = sample(logits, **kwargs)
    return next.to(dtype=x.dtype)

@torch.inference_mode()
def generate(
    model,
    prompt: torch.Tensor,
    max_returned_tokens: int,
    *,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.

    The implementation of this function is modified from A. Karpathy's nanoGPT.

    Args:
        model: The model to use.
        prompt: Tensor of shape (T) with indices of the prompt sequence.
        max_returned_tokens: The maximum number of tokens to return (given plus generated).
        temperature: Scales the predicted logits by 1 / temperature.
        top_k: If specified, only sample among the tokens with the k highest probabilities.
        eos_id: If specified, stop generating any more token once the <eos> token is triggered.
    """
    T = prompt.size(0)
    assert max_returned_tokens > T
    if model.max_seq_length < max_returned_tokens - 1:
        # rolling the kv cache based on the `input_pos` value would be necessary. However, doing so would introduce a
        # data dependency on the `input_pos` tensor and impact model compilation. Since this setting is uncommon, we do
        # not support it to avoid negatively impacting the overall speed
        raise NotImplementedError(f"max_seq_length {model.max_seq_length} needs to be >= {max_returned_tokens - 1}")

    tokens = [prompt]
    for _ in range(2, max_returned_tokens - T + 1):
        token = next_token(model, prompt.view(1, -1), temperature=temperature, top_k=top_k).clone()
        tokens.append(token)
        prompt = torch.cat(tokens)
        if token == eos_id:
            break
    return torch.cat(tokens)

if __name__ == "__main__":
    from jsonargparse import CLI

    torch.set_float32_matmul_precision("high")
    CLI(main)
