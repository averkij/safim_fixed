import os
import hashlib
import json
import re
from pathlib import Path
from argparse import Namespace

import openai
import tiktoken
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, NoBadWordsLogitsProcessor


class ModelWrapper:
    max_new_tokens = 128
    seed = 0

    def count_prompt_tokens(self, prompt):
        return len(self.tokenizer.encode(prompt, add_special_tokens=True))

    def check_context(self, input_tokens):
        if input_tokens + self.max_new_tokens > self.max_length:
            raise ValueError("Prompt plus requested output exceeds context limit; use structured FIM trimming or a larger declared context")

    def fit_infilling_prompt(self, prefix, suffix, reverse=False):
        """Keep context nearest the hole; assemble sentinels after trimming.

        Trim characters (never partial Unicode bytes) from the longer side,
        checking the actual assembled token count after each reduction.
        """
        original = (len(prefix), len(suffix))
        while True:
            prompt = self.assemble_infilling_prompt(prefix, suffix, reverse)
            count = self.count_prompt_tokens(prompt)
            if count + self.max_new_tokens <= self.max_length:
                self.last_context = dict(input_tokens=count, max_new_tokens=self.max_new_tokens,
                    context_limit=self.max_length, removed_prefix_chars=original[0] - len(prefix),
                    removed_suffix_chars=original[1] - len(suffix))
                return prompt
            if not prefix and not suffix:
                raise ValueError("FIM markers and output budget do not fit the context limit")
            if len(prefix) >= len(suffix):
                prefix = prefix[max(1, len(prefix) // 8):]
            else:
                suffix = suffix[:-max(1, len(suffix) // 8)]

    def invoke(self, prompt: str) -> str:
        raise NotImplementedError()

    def invoke_cached(self, prompt: str, cache: dict) -> str:
        identity = {
            "protocol": 2,
            "wrapper": type(self).__name__,
            "model": getattr(self, "model_name", None),
            "configuration": getattr(self, "cache_configuration", {}),
            "max_new_tokens": self.max_new_tokens,
            "context_limit": getattr(self, "max_length", None),
            "seed": self.seed,
            "prompt": prompt,
        }
        key = "safim-v2:" + hashlib.sha256(json.dumps(
            identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
        if key not in cache:
            # Per-prompt seeding makes cache hits and dataset iteration order
            # irrelevant to the RNG stream used for this completion.
            self.request_seed = int.from_bytes(hashlib.sha256(
                json.dumps([self.seed, prompt], ensure_ascii=False).encode("utf-8")
            ).digest()[:4], "big") % (2 ** 31)
            transformers.set_seed(self.request_seed)
            answer = self.invoke(prompt)
            if not isinstance(answer, str):
                raise TypeError("Generation must return text")
            cache[key] = answer
        if not isinstance(cache[key], str):
            raise ValueError("Cached completion must be text")
        return cache[key]

    def assemble_completion_prompt(self, prompt: str) -> str:
        return prompt

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        raise NotImplementedError()


class CodeLlama(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        self.max_length = max_length
        assert model_name.startswith("codellama/CodeLlama")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_name,
            tokenizer=self.tokenizer,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(["#", "▁#", "/*", "▁/*"])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        self.check_context(self.count_prompt_tokens(prompt))
        generated_text = self.pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            temperature=0.2,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            top_p=0.95,
            logits_processor=self.logits_processor
        )[0]["generated_text"]
        return generated_text[len(prompt):]

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return "<PRE>" + " <SUF>" + suffix + " <MID>" + prefix
        else:
            return "<PRE>" + prefix + " <SUF>" + suffix + " <MID>"


class Incoder(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("facebook/incoder")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        device = torch.device("cuda")
        if model_name.endswith("6B"):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                revision="float16",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(
                        ['#Ġ', '/*Ġ', 'Ġ#Ġ', 'ĠĠ#Ġ', 'ĠĠ/*Ġ', 'ĠĠĠ#Ġ', 'ĠĠĠ/*Ġ', 'ĠĠĠĠ#Ġ', 'ĠĠĠĠ/*Ġ', 'ĠĠĠĠĠ#Ġ',
                         'ĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠ/*Ġ', 'ĠĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠĠĠ/*Ġ', 'ĠĠĠĠĠĠĠĠĠ#Ġ',
                         'ĠĠĠĠĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠĠĠĠĠĠĠĠ#Ġ', 'ĠĠĠĠĠĠĠĠĠĠĠĠĠĠ#Ġ',
                         'ĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠĠ#Ġ']
                    )],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids.to(self.model.device)
        input_ids_len = input_ids.shape[1]
        self.check_context(input_ids_len)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.2,
                max_new_tokens=self.max_new_tokens,
                top_p=0.95,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:], clean_up_tokenization_spaces=False
        )
        EOM = "<|endofmask|>"
        if EOM not in generated_text:
            generated_text += EOM
        return generated_text[:generated_text.index(EOM)]

    @staticmethod
    def make_sentinel(i):
        # signals (1) a location to insert an infill and (2) the start of the infill generation
        return f"<|mask:{i}|>"

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return prefix + self.make_sentinel(0) + suffix + self.make_sentinel(1) + self.make_sentinel(0)
        else:
            return self.make_sentinel(0) + suffix + self.make_sentinel(1) + self.make_sentinel(0) + prefix


class OpenAIModel(ModelWrapper):
    def __init__(self, model_name, max_length, timeout=120.0, max_retries=2):
        if timeout <= 0 or max_retries < 0:
            raise ValueError("API timeout must be positive and retries nonnegative")
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.client = openai.OpenAI(timeout=timeout, max_retries=max_retries)

    def count_prompt_tokens(self, prompt):
        # max_length already excludes the declared chat overhead allowance.
        return len(self.tokenizer.encode_ordinary(prompt))

    def invoke(self, prompt: str) -> str:
        self.check_context(self.count_prompt_tokens(prompt))
        result = self.client.chat.completions.create(
            model=self.model_name,
            max_completion_tokens=self.max_new_tokens,
            temperature=0.2,
            top_p=0.95,
            messages=[{"role": "user", "content": prompt}],
            seed=getattr(self, "request_seed", self.seed),
        )
        content = result.choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("API returned no text completion")
        return content

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return suffix + "\n\n" + prefix
        else:
            raise NotImplementedError()


class CodegenModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("Salesforce/codegen")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        device = torch.device("cuda")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = 50256
        if "6B" in model_name:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            ).to(device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(['#', 'Ġ#', '/*', 'Ġ/*'])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids.to(self.model.device)
        input_ids_len = input_ids.shape[1]
        self.check_context(input_ids_len)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.2,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=50256,
                top_p=0.95,
                use_cache=True,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:],
            skip_special_tokens=True
        )
        return generated_text

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return suffix + "\n\n" + prefix
        else:
            raise NotImplementedError()


class StarcoderModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("bigcode/starcoder")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.max_length = max_length
        device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(['#', 'Ġ#', '/*', 'Ġ/*'])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids.to(self.model.device)
        input_ids_len = input_ids.shape[1]
        self.check_context(input_ids_len)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.2,
                max_new_tokens=self.max_new_tokens,
                top_p=0.95,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:],
            skip_special_tokens=True
        )
        return generated_text

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return "<fim_prefix>" + "<fim_suffix>" + suffix + "<fim_middle>" + prefix
        else:
            return "<fim_prefix>" + prefix + "<fim_suffix>" + suffix + "<fim_middle>"


class DeepseekModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("deepseek-ai/deepseek")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.max_length = max_length
        device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        ).to(device)
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(['#', 'Ġ#', '/*', 'Ġ/*'])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids.to(self.model.device)
        input_ids_len = input_ids.shape[1]
        self.check_context(input_ids_len)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.2,
                max_new_tokens=self.max_new_tokens,
                top_p=0.95,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:],
            skip_special_tokens=True
        )
        return generated_text

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return "<｜fim▁begin｜>" + "<｜fim▁hole｜>" + suffix + "<｜fim▁end｜>" + prefix
        else:
            return "<｜fim▁begin｜>" + prefix + "<｜fim▁hole｜>" + suffix + "<｜fim▁end｜>"


class PhiModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("microsoft/phi")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.max_length = max_length
        device = torch.device("cuda")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token_id = 50256
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, trust_remote_code=True
        ).to(device)
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(['#', 'Ġ#', '/*', 'Ġ/*'])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, truncation=False, return_attention_mask=False, return_tensors="pt"
        ).input_ids.to(self.model.device)
        input_ids_len = input_ids.shape[1]
        self.check_context(input_ids_len)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.2,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=50256,
                top_p=0.95,
                use_cache=True,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:],
            skip_special_tokens=True
        )
        return generated_text

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return suffix + "\n\n" + prefix
        else:
            raise NotImplementedError()


class MixtralModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("mistralai/Mixtral")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.max_length = max_length
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(['▁#', '▁/*'])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, truncation=False, return_attention_mask=False, return_tensors="pt"
        ).input_ids.to(self.model.device)
        input_ids_len = input_ids.shape[1]
        self.check_context(input_ids_len)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.2,
                max_new_tokens=self.max_new_tokens,
                top_p=0.95,
                use_cache=True,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:],
            skip_special_tokens=True
        )
        return generated_text

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return suffix + "\n\n" + prefix
        else:
            raise NotImplementedError()


class WizardModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("WizardLM/WizardCoder")
        if model_name == "WizardLM/WizardCoder-33B-V1.1":
            self.use_deepseek_base = True
        else:
            self.use_deepseek_base = False
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.max_length = max_length
        device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16
        ).to(device)
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(['#', 'Ġ#', '/*', 'Ġ/*'])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, truncation=False, return_attention_mask=False, return_tensors="pt"
        ).input_ids.to(self.model.device)
        input_ids_len = input_ids.shape[1]
        self.check_context(input_ids_len)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.2,
                max_new_tokens=self.max_new_tokens,
                top_p=0.95,
                use_cache=True,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:],
            skip_special_tokens=True
        )
        return generated_text

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if self.use_deepseek_base:
            if reverse:
                return "<｜fim▁begin｜>" + "<｜fim▁hole｜>" + suffix + "<｜fim▁end｜>" + prefix
            else:
                return "<｜fim▁begin｜>" + prefix + "<｜fim▁hole｜>" + suffix + "<｜fim▁end｜>"
        else:
            if reverse:
                return "<fim_prefix>" + "<fim_suffix>" + suffix + "<fim_middle>" + prefix
            else:
                return "<fim_prefix>" + prefix + "<fim_suffix>" + suffix + "<fim_middle>"


class SantacoderModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        assert model_name.startswith("bigcode/santacoder")
        if ":" in model_name:
            model_name, revision = model_name.split(":")
        else:
            revision = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.max_length = max_length
        device = torch.device("cuda")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(device)
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(['#', 'Ġ#', '/*', 'Ġ/*'])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        input_ids = self.tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids.to(self.model.device)
        input_ids_len = input_ids.shape[1]
        self.check_context(input_ids_len)
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                do_sample=True,
                num_return_sequences=1,
                temperature=0.2,
                max_new_tokens=self.max_new_tokens,
                top_p=0.95,
                pad_token_id=self.tokenizer.unk_token_id,
                logits_processor=self.logits_processor
            )
        generated_text = self.tokenizer.decode(
            generated_ids[0, input_ids_len:],
            skip_special_tokens=True
        )
        return generated_text

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return "<fim-prefix>" + "<fim-suffix>" + suffix + "<fim-middle>" + prefix
        else:
            return "<fim-prefix>" + prefix + "<fim-suffix>" + suffix + "<fim-middle>"


class MagicCoderModel(ModelWrapper):
    def __init__(self, model_name, max_length, block_comments=False):
        self.max_length = max_length
        assert model_name.startswith("ise-uiuc/Magicoder")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.model_max_length = max_length
        self.pipeline = transformers.pipeline(
            model=model_name,
            tokenizer=self.tokenizer,
            task="text-generation",
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.logits_processor = (
            [
                NoBadWordsLogitsProcessor(
                    [[word_idx] for word_idx in self.tokenizer.convert_tokens_to_ids(['#', 'Ġ#', '/*', 'Ġ/*'])],
                    self.tokenizer.eos_token_id
                )
            ]
            if block_comments
            else None
        )

    def invoke(self, prompt: str) -> str:
        self.check_context(self.count_prompt_tokens(prompt))
        generated_text = self.pipeline(
            prompt,
            do_sample=True,
            num_return_sequences=1,
            temperature=0.2,
            max_new_tokens=self.max_new_tokens,
            top_p=0.95,
            logits_processor=self.logits_processor
        )[0]["generated_text"]
        return generated_text[len(prompt):]

    def assemble_infilling_prompt(self, prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return suffix + "\n\n" + prefix
        else:
            raise NotImplementedError()


NATIVE_FIM_MODELS = {
    "Qwen/Qwen2.5-Coder-3B": dict(revision="09d9bc5d376b0cfa0100a0694ea7de7232525803",
        markers=("<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"), order="psm"),
    "JetBrains/Mellum-4b-base": dict(revision="83cce2605fbdf6a3868627e9b0a5924e0072b94d",
        markers=("<fim_prefix>", "<fim_suffix>", "<fim_middle>"), order="spm"),
}


class NativeFIMModel(ModelWrapper):
    """Explicit pinned base-model protocols, independent of legacy adapters."""
    def __init__(self, model_name, revision=None, context_limit=None, device=None, dtype=None, block_comments=False):
        if block_comments:
            raise ValueError("Comment-token banning is not defined for the native FIM protocol")
        protocol = NATIVE_FIM_MODELS[model_name]
        revision = revision or protocol["revision"]
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("Native FIM model revision must be an immutable 40-character commit")
        device, dtype = device or "cuda", dtype or "float16"
        if device not in {"cpu", "cuda"} or dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("Unsupported device/dtype")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, trust_remote_code=False)
        config = transformers.AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=False)
        native_limit = config.max_position_embeddings
        self.max_length = context_limit if context_limit is not None else native_limit
        if not 0 < self.max_length <= native_limit:
            raise ValueError("Context limit must be positive and within the model configuration")
        self.markers, self.fim_order = protocol["markers"], protocol["order"]
        for marker in self.markers:
            ids = self.tokenizer.encode(marker, add_special_tokens=False)
            if len(ids) != 1 or ids[0] == self.tokenizer.unk_token_id:
                raise ValueError(f"Tokenizer does not define the expected FIM marker: {marker}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, config=config,
            trust_remote_code=False, torch_dtype=getattr(torch, dtype), low_cpu_mem_usage=True).to(device).eval()
        self.protocol_configuration = dict(model_revision=revision, tokenizer_revision=revision,
            fim_order=self.fim_order, fim_markers=list(self.markers), context_limit=self.max_length,
            device=device, dtype=dtype, temperature=0.2, top_p=0.95, top_k=0, repetition_penalty=1.0)

    def count_prompt_tokens(self, prompt):
        return len(self.tokenizer.encode(prompt, add_special_tokens=False))

    def assemble_infilling_prompt(self, prefix, suffix, reverse=False):
        if reverse:
            raise ValueError("reverse_infilling is not defined for the native model-card protocol")
        pre, suf, mid = self.markers
        if self.fim_order == "psm":
            return pre + prefix + suf + suffix + mid
        return suf + suffix + pre + prefix + mid

    def invoke(self, prompt):
        encoded = self.tokenizer(prompt, add_special_tokens=False, truncation=False,
            return_tensors="pt", return_token_type_ids=False).to(self.model.device)
        length = encoded["input_ids"].shape[1]
        self.check_context(length)
        with torch.no_grad():
            generated = self.model.generate(**encoded, do_sample=True, temperature=0.2,
                top_p=0.95, top_k=0, repetition_penalty=1.0, num_return_sequences=1,
                max_new_tokens=self.max_new_tokens, eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
        return self.tokenizer.decode(generated[0, length:], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)


def build_model(args: Namespace) -> ModelWrapper:
    max_new_tokens = getattr(args, "max_new_tokens", 128)
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    native_options = {key: getattr(args, key, None) for key in ("model_revision", "context_limit", "device", "dtype")}
    if args.model_name not in NATIVE_FIM_MODELS and any(value is not None for value in native_options.values()):
        raise ValueError("Explicit revision/context/device/dtype options require a registered native FIM adapter")
    if args.model_name in NATIVE_FIM_MODELS:
        model_wrapper = NativeFIMModel(args.model_name, revision=native_options["model_revision"],
            context_limit=native_options["context_limit"], device=native_options["device"],
            dtype=native_options["dtype"], block_comments=args.block_comments)
    elif args.model_name.startswith("codellama/CodeLlama"):
        model_wrapper = CodeLlama(args.model_name, 4096, args.block_comments)
    elif args.model_name.startswith("facebook/incoder"):
        model_wrapper = Incoder(args.model_name, 2048, args.block_comments)
    elif args.model_name.startswith("gpt-"):
        model_wrapper = OpenAIModel(args.model_name, 4096 - 32,
                                    timeout=getattr(args, "api_timeout", 120.0),
                                    max_retries=getattr(args, "api_max_retries", 2))
    elif args.model_name.startswith("Salesforce/codegen"):
        model_wrapper = CodegenModel(args.model_name, 2048, args.block_comments)
    elif args.model_name.startswith("bigcode/starcoder"):
        model_wrapper = StarcoderModel(args.model_name, 2048, args.block_comments)
    elif args.model_name.startswith("deepseek-ai/deepseek"):
        model_wrapper = DeepseekModel(args.model_name, 4096, args.block_comments)
    elif args.model_name.startswith("microsoft/phi"):
        model_wrapper = PhiModel(args.model_name, 2048, args.block_comments)
    elif args.model_name.startswith("mistralai/Mixtral"):
        model_wrapper = MixtralModel(args.model_name, 4096, args.block_comments)
    elif args.model_name.startswith("WizardLM/WizardCoder"):
        model_wrapper = WizardModel(args.model_name, 2048, args.block_comments)
    elif args.model_name.startswith("bigcode/santacoder"):
        model_wrapper = SantacoderModel(args.model_name, 2048, args.block_comments)
    elif args.model_name.startswith("ise-uiuc/Magicoder"):
        model_wrapper = MagicCoderModel(args.model_name, 4096, args.block_comments)
    else:
        raise ValueError(args.model_name)
    model_wrapper.max_new_tokens = max_new_tokens
    model_wrapper.seed = getattr(args, "seed", 0)
    model_wrapper.model_name = args.model_name
    # Conservative identity: changed protocol/code/configuration never reuses a
    # completion from another model or experiment. Old prompt-only keys miss.
    excluded = {"cache_path", "output_path", "checkpoint_every"}
    model_wrapper.cache_configuration = {key: value for key, value in vars(args).items()
                                         if key not in excluded}
    protocol_configuration = getattr(model_wrapper, "protocol_configuration", {})
    if isinstance(protocol_configuration, dict):
        model_wrapper.cache_configuration.update(protocol_configuration)
    model_wrapper.cache_configuration["wrapper_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()).hexdigest()
    tokenizer_config = getattr(getattr(model_wrapper, "tokenizer", None), "init_kwargs", {})
    if isinstance(tokenizer_config, dict):
        model_wrapper.cache_configuration["resolved_tokenizer_revision"] = tokenizer_config.get("_commit_hash")
    model_config = getattr(getattr(model_wrapper, "model", None), "config", None)
    revision = getattr(model_config, "_commit_hash", None)
    if isinstance(revision, str):
        model_wrapper.cache_configuration["resolved_model_revision"] = revision
    return model_wrapper
