import unittest
from unittest.mock import Mock
from model_support import load_models, wrapper


class GenerationRuntimeFailureTests(unittest.TestCase):
    def test_deepseek_runtime_failure_not_cached(self):
        model = wrapper(load_models(), "DeepseekModel")
        model.model.generate.side_effect = RuntimeError("CUDA out of memory (synthetic)")
        cache = {}
        with self.assertRaises(RuntimeError):
            model.invoke_cached("prompt", cache)
        self.assertEqual(cache, {})

    def test_successful_answer_is_cached(self):
        model = wrapper(load_models(), "DeepseekModel")
        cache = {}
        self.assertEqual(model.invoke_cached("prompt", cache), "completion")
        self.assertEqual(model.invoke_cached("prompt", cache), "completion")
        self.assertEqual(model.model.generate.call_count, 1)

    def test_legitimate_empty_answer_is_cached(self):
        model = wrapper(load_models(), "DeepseekModel")
        model.tokenizer.decode = Mock(return_value="")
        cache = {}
        self.assertEqual(model.invoke_cached("prompt", cache), "")
        self.assertEqual(len(cache), 1)


if __name__ == "__main__":
    unittest.main()
