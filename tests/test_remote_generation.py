import os
import unittest
from unittest.mock import patch
from model_support import load_models, wrapper


class RemoteGenerationTests(unittest.TestCase):
    def test_explicit_request_and_text(self):
        model = wrapper(load_models(), "OpenAIModel")
        model.max_new_tokens = 256
        self.assertEqual(model.invoke("prompt"), "completion")
        call = model.client.chat.completions.create.call_args
        self.assertIsNotNone(call)
        self.assertEqual(call.kwargs, dict(model="gpt-fixture", max_completion_tokens=256,
            temperature=0.2, top_p=0.95, seed=0, messages=[dict(role="user", content="prompt")]))

    def test_failed_request_propagates_without_cache_entry(self):
        model = wrapper(load_models(), "OpenAIModel")
        model.client.chat.completions.create.side_effect = RuntimeError("synthetic transport failure")
        cache = {}
        with self.assertRaises(RuntimeError):
            model.invoke_cached("prompt", cache)
        self.assertEqual(cache, {})
        self.assertEqual(model.client.chat.completions.create.call_count, 1)

    def test_missing_text_is_not_success(self):
        model = wrapper(load_models(), "OpenAIModel")
        model.client.chat.completions.create.return_value.choices[0].message.content = None
        with self.assertRaises(RuntimeError):
            model.invoke("prompt")

    def test_sdk_owns_bounded_retries(self):
        module = load_models()
        with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-not-a-key"}):
            module.OpenAIModel("gpt-fixture", 1000)
        module.openai.OpenAI.assert_called_once_with(timeout=120.0, max_retries=2)

    def test_configurable_limits(self):
        module = load_models()
        module.OpenAIModel("gpt-fixture", 1000, timeout=15, max_retries=0)
        module.openai.OpenAI.assert_called_once_with(timeout=15, max_retries=0)


if __name__ == "__main__":
    unittest.main()
