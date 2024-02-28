# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import sys
from pathlib import Path
from typing import Dict, List, Literal, Optional

import argparse
import random
import re
import sys
import torch
import warnings
# from transformers import AutoTokenizer, pipeline
from tqdm import tqdm, trange
from tqdm.contrib import tenumerate

import lightning as L
import torch
from lightning.fabric.plugins import BitsandbytesPrecision
from lm_eval import base, evaluator, tasks
from lm_eval.base import BaseLM

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt.config import Config
# from lit_gpt.model import GPT
from lit_gpt.model import GPT
from lit_gpt import Tokenizer
#from lit_gpt import GPT, Config, Tokenizer
from lit_gpt.utils import check_valid_checkpoint_dir, get_default_supported_precision, load_checkpoint



class EvalHarnessBase(BaseLM):
    # Credits:
    # https://github.com/EleutherAI/gpt-neox/blob/main/eval_tasks/eval_adapter.py
    def __init__(self, fabric: L.Fabric, model: GPT, tokenizer: Tokenizer, batch_size: int):
        super().__init__()
        self.fabric = fabric
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size_per_gpu = batch_size
        with fabric.init_tensor():
            model.set_kv_cache(batch_size=batch_size)

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        kwargs = {el.split("=")[0]: el.split("=")[1] for el in arg_string.split(",")}
        return cls(**kwargs, **additional_config)

    @property
    def eot_token_id(self):
        # we use EOT because end of *text* is more accurate for what we're doing than end of *sentence*
        return self.tokenizer.eos_id

    @property
    def max_length(self):
        return self.model.max_seq_length

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self.batch_size_per_gpu * self.fabric.world_size

    @property
    def device(self):
        return self.fabric.device

    def tok_encode(self, string: str) -> List[int]:
        return self.tokenizer.encode(string, bos=False, eos=False).tolist()

    def tok_decode(self, tokens: List[int]) -> str:
        t = torch.tensor(tokens)
        return self.tokenizer.decode(t)

    @torch.inference_mode()
    def _model_call(self, inps):
        return self.model(inps)

    @torch.inference_mode()
    def _model_generate(self, context, max_length,
                        eos_token_id) -> torch.Tensor:
        # this only supports batch size 1
        assert context.shape[0] == 1
        out = generate(self.model, context[0], max_length, eos_id=eos_token_id)
        for block in self.model.transformer.h:
            block.attn.kv_cache.reset_parameters()
        return out.unsqueeze(0)
    
def generate_prompt(n_garbage):
    """Generates a text file and inserts an execute line at a random position."""
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 10000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key
@torch.no_grad()
def test_model(fabric, model, tokenizer, prompt_text,pass_key) -> torch.Tensor:
    # # response = pipe(prompt_text, num_return_sequences=1, max_new_tokens=10)[
    # #     0]["generated_text"][len(prompt_text):]
    # # response =  eval_harness._model_generate
    # response = generate(model, eval_harness.tok_encode(prompt_text), len(prompt_text), eos_id=eval_harness.eot_token_id)
    encoded = tokenizer.encode(prompt_text, device=fabric.device)
    with fabric.init_tensor():
        # do not set `max_seq_length=max_returned_token` because memory is not a concern here
        model.set_kv_cache(batch_size=1)
    output = generate(
        model, encoded, max_returned_tokens=len(encoded)+10, temperature=0.8, eos_id=tokenizer.eos_id
    )
    model.clear_kv_cache()
    output = tokenizer.decode(output)
    # fabric.print(output)
    response = output
    assert f"The pass key is {pass_key}" in prompt_text

    try:
        pass_key = int(re.search(r'\d+', response).group())
    except:
        pass_key = response[:20]

    return pass_key
@torch.inference_mode()
def run_eval_harness(
    checkpoint_dir: Path,
    model_file: Optional[str] = None,
    tokenizer_dir: Optional[Path] = None,
    config_filepath: Optional[Path] = None,
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]] = None,
    # eval_tasks: List[str] = [
    #     "arc_challenge", "piqa", "hellaswag", "hendrycksTest-*"
    # ],
    save_filepath: Optional[Path] = None,
    num_fewshot: int = 0,
    limit: Optional[int] = None,
    bootstrap_iters: int = 100000,
    no_cache: bool = True,
    fixed_length:Optional[int] = None,
    max_tokens:Optional[int] = 8192,
    min_tokens:Optional[int] = 256,
    tokens_step:Optional[int] = None,
    length_step:Optional[int] = 128,
    iterations:Optional[int] = 20,
    # fixed_length:Optional[int] = None,
):
    if precision is None:
        precision = get_default_supported_precision(training=False)
    plugins = None
    if quantize is not None and quantize.startswith("bnb."):
        if "mixed" in precision:
            raise ValueError(
                "Quantization and mixed precision is not supported.")
        dtype = {
            "16-true": torch.float16,
            "bf16-true": torch.bfloat16,
            "32-true": torch.float32
        }[precision]
        plugins = BitsandbytesPrecision(quantize[4:], dtype)
        precision = None

    fabric = L.Fabric(devices=1, precision=precision, plugins=plugins)
    if tokenizer_dir is None:
        tokenizer_dir = checkpoint_dir
    check_valid_checkpoint_dir(tokenizer_dir)
    tokenizer = Tokenizer(tokenizer_dir)

    if config_filepath is None:
        config_filepath = checkpoint_dir / "lit_config.json"
    config = Config.from_json(config_filepath)

    if model_file is None:
        if quantize == "gptq.int4":
            model_file = "lit_model_gptq.4bit.pth"
            if not (checkpoint_dir / model_file).is_file():
                raise ValueError("Please run `python quantize/gptq.py` first")
        else:
            model_file = "lit_model.pth"
    checkpoint_path = checkpoint_dir / model_file

    print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}",
          file=sys.stderr)
    with fabric.init_module(empty_init=True):
        model = GPT(config)

    model.eval()
    model = fabric.setup_module(model)
    try:
        load_checkpoint(fabric, model, checkpoint_path, strict=True)
    except Exception:
        print(
            "Failed to load checkpoint with strict=True, trying again with strict=False"
        )
        print("error message details:", file=sys.stderr)
        import traceback
        traceback.print_exc()
        load_checkpoint(fabric, model, checkpoint_path, strict=False)
    # tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m-v0", model_max_length=sys.maxsize, padding_side="right", trust_remote_code=True)
    # eval_harness = EvalHarnessBase(fabric, model, tokenizer, 1)
    if fixed_length:
        lengths = [fixed_length]
        tokens = [len(tokenizer.encode(generate_prompt(fixed_length)[0]))]
        print(f"Prompt is {tokens[0]} tokens")
    else:
        if tokens_step:
            tokens = [x for x in range(
                min_tokens, max_tokens + 1, tokens_step)]
        else:
            tokens = [min_tokens]
            while min_tokens < max_tokens:
                point = tokens[-1] * 2
                if point <= max_tokens:
                    tokens.append(point)
                else:
                    break

        lengths = []
        last_n = 0
        for target in tqdm(tokens, desc="Determining sequence lengths"):
            num_tokens = 0
            n = last_n
            while num_tokens < target:
                last_n = n
                n += length_step
                prompt = generate_prompt(n)[0]
                num_tokens = len(tokenizer.encode(prompt))
            lengths.append(last_n)
    results = []
    torch.cuda.empty_cache()
    # loaded = load_model_and_apply_patches(model, args)
    # pipe = pipeline("text-generation", model=model,
    #                     tokenizer=tokenizer, pad_token_id=eos_token_id)
    #pipe = pipeline("text-generation", model="EleutherAI/pythia-160m-v0",tokenizer=tokenizer,pad_token_id=tokenizer.eos_id)
    #pipe = pipeline("text-generation", model="EleutherAI/pythia-160m-v0",tokenizer=tokenizer,pad_token_id=tokenizer.eos_id)
    result = [0] * len(lengths)
    for i, length in tenumerate(lengths, desc="Lengths", leave=False):
        for _ in trange(0, iterations, desc="Iterations", leave=False):
            prompt_text, pass_key = generate_prompt(length)

            # num_tokens = len(pipe.tokenizer.encode(prompt_text))
            num_tokens = len(tokenizer.encode(prompt_text))
            answer = test_model(fabric, model, tokenizer, prompt_text,pass_key)
            if answer == pass_key:
                result[i] += 1
        result[i] /= iterations
        print(f"{0}: {tokens[i]}={int(result[i]*100)}%")

    result.insert(0, "model")
    results.append(result)

    if save_filepath:
        with open(save_filepath, "w", encoding="utf-8") as f:
            f.write(f",{','.join([str(x) for x in tokens])}\n")
            for result in results:
                f.write(f"{','.join([str(x) for x in result])}\n")
if __name__ == "__main__":
    from jsonargparse import CLI
    # warnings.simplefilter("ignore")
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-m", "--model", action="append", nargs="+")
    # parser.add_argument("--fixed-length", type=int)
    # parser.add_argument("--max-tokens", type=int, default=8192)
    # parser.add_argument("--min-tokens", type=int, default=256)
    # parser.add_argument("--tokens-step", type=int)
    # parser.add_argument("--length-step", type=int, default=128)
    # parser.add_argument("--iterations", type=int, default=20)
    # parser.add_argument("--output-file", type=str)
    # args = parser.parse_args()
    # main(add_args(parser).parse_args())
    CLI(run_eval_harness, as_positional=False)