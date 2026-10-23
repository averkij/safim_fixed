"""Exercise the real evaluator without loading execution-only dependencies."""

import pathlib
import types
import unittest
from unittest.mock import patch


EVALUATE_SOURCE = (pathlib.Path(__file__).resolve().parents[1] / "evaluate.py").read_text(encoding="utf-8")


def load_evaluator(source):
    # These imports are unrelated to API matching. Fail loudly if a test uses
    # execution or dataset machinery, rather than silently mocking its result.
    def unavailable(*args, **kwargs):
        raise AssertionError("Execution/dataset dependencies are outside this unit test")

    dependencies = {}
    for name, attributes in {
        "ast_utils": ["ErrorCheckVisitor", "get_parser"],
        "data_utils": ["load_dataset", "stream_jsonl"],
        "exec_utils": ["build_execeval", "run_test"],
        "tqdm": ["tqdm"],
    }.items():
        module = types.ModuleType(name)
        for attribute in attributes:
            setattr(module, attribute, unavailable)
        if name == "exec_utils":
            module.ExecEvalError = RuntimeError
        dependencies[name] = module
    evaluator = types.ModuleType("safim_api_scoring_under_test")
    with patch.dict("sys.modules", dependencies):
        exec(compile(source, "evaluate.py", "exec"), evaluator.__dict__)
    return evaluator


class ApiScoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator(EVALUATE_SOURCE)

    def match(self, candidate, reference, language="python"):
        return self.evaluator.syntax_match(candidate, reference, language)

    def test_different_function(self):
        self.assertFalse(self.match("np.zeros((n,n))", "np.ones((n,n))"))

    def test_zero_argument_function(self):
        self.assertFalse(self.match("x.y()", "plt.show()"))

    def test_different_receiver(self):
        self.assertFalse(self.match("left.save(x)", "right.save(x)"))

    def test_nested_callable(self):
        self.assertFalse(self.match("factory_a()(x)", "factory_b()(x)"))
        self.assertTrue(self.match("factory_a()(x)", "factory_a()(x)"))

    def test_subscript_callable(self):
        self.assertFalse(self.match("handlers[0](x)", "handlers[1](x)"))

    def test_identical_calls(self):
        self.assertTrue(self.match("np.ones((n,n))", "np.ones((n,n))"))
        self.assertTrue(self.match("plt.show()", "plt.show()"))

    def test_equivalent_formatting(self):
        self.assertTrue(self.match("np.ones( (n, n) )", "np.ones((n,n))"))

    def test_different_arguments(self):
        self.assertFalse(self.match("f(x)", "f(y)"))
        self.assertFalse(self.match("f(x=1)", "f(x=2)"))

    def test_keyword_order_preserved(self):
        self.assertTrue(self.match("f(a=1,b=2)", "f(b=2,a=1)"))

    def test_other_language_behavior_preserved(self):
        self.assertTrue(self.match("obj.f( x )", "obj.f(x)", "java"))
        self.assertFalse(self.match("obj.g(x)", "obj.f(x)", "java"))

    def test_non_call_and_parse_failure_fallback_preserved(self):
        self.assertTrue(self.match("x + y", "x+y"))
        self.assertFalse(self.match("x+y", "x-y"))
        self.assertTrue(self.match("f(", "f("))
        self.assertFalse(self.match("f(", "g("))


if __name__ == "__main__":
    unittest.main()
