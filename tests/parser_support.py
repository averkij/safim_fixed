import sys
import types
from pathlib import Path
from source_support import source_at


def load_ast():
    module = types.ModuleType("ast_under_test")
    module.__file__ = str(Path(__file__).resolve().parents[1] / "ast_utils.py")
    exec(compile(source_at("ast_utils.py"), "ast_utils.py", "exec"), module.__dict__)
    return module


def load_prompt():
    from unittest.mock import patch
    dependencies = {"ast_utils": load_ast(), "model_utils": types.ModuleType("model_utils")}
    dependencies["model_utils"].ModelWrapper = object
    module = types.ModuleType("prompt_under_test")
    with patch.dict(sys.modules, dependencies):
        exec(compile(source_at("prompt_utils.py"), "prompt_utils.py", "exec"), module.__dict__)
    return module
