"""Load model wrappers with lightweight runtime doubles; no model downloads."""
import contextlib
import sys
import types
from unittest.mock import Mock, patch
from source_support import source_at


class Tensor:
    shape = (1, 10)
    def to(self, *_):
        return self
    def __getitem__(self, _):
        return self


class Tokenizer:
    eos_token_id = 0
    unk_token_id = 0
    def __call__(self, *args, **kwargs):
        return types.SimpleNamespace(input_ids=Tensor())
    def encode_ordinary(self, text):
        return list(text)
    def decode(self, *args, **kwargs):
        return "completion"


def load_models():
    dependencies = {name: types.ModuleType(name) for name in ["torch", "transformers", "openai", "tiktoken"]}
    dependencies["torch"].no_grad = contextlib.nullcontext
    for name in ["AutoModelForCausalLM", "AutoTokenizer", "NoBadWordsLogitsProcessor"]:
        setattr(dependencies["transformers"], name, Mock())
    dependencies["openai"].ChatCompletion = types.SimpleNamespace(create=Mock(return_value={"choices": [{"message": {"content": "completion"}}]}))
    module = types.ModuleType("models_under_test")
    with patch.dict(sys.modules, dependencies):
        exec(compile(source_at("model_utils.py"), "model_utils.py", "exec"), module.__dict__)
    return module


def wrapper(module, name):
    instance = getattr(module, name).__new__(getattr(module, name))
    instance.tokenizer = Tokenizer()
    instance.max_length = 4096
    instance.logits_processor = None
    instance.model_name = "gpt-fixture"
    instance.model = types.SimpleNamespace(device="cpu", generate=Mock(return_value=Tensor()))
    instance.pipeline = Mock(return_value=[{"generated_text": "promptcompletion"}])
    return instance
