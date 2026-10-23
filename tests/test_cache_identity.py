import unittest
from unittest.mock import Mock
from model_support import load_models, wrapper


class CacheIdentityTests(unittest.TestCase):
    def model(self, name="model-a", budget=128, **configuration):
        model = wrapper(load_models(), "StarcoderModel")
        model.model_name = name
        model.max_new_tokens = budget
        model.cache_configuration = configuration
        model.invoke = Mock(return_value=name)
        return model

    def test_different_models_do_not_share_answers(self):
        cache = {}
        self.assertEqual(self.model("a").invoke_cached("prompt", cache), "a")
        self.assertEqual(self.model("b").invoke_cached("prompt", cache), "b")
        self.assertEqual(len(cache), 2)

    def test_changed_budget_seed_or_revision_misses(self):
        for changed in [dict(budget=256), dict(seed=1), dict(revision="new")]:
            with self.subTest(changed=changed):
                cache = {}
                self.model().invoke_cached("prompt", cache)
                other = self.model(**changed)
                other.invoke_cached("prompt", cache)
                other.invoke.assert_called_once_with("prompt")

    def test_legacy_prompt_cache_is_not_trusted(self):
        cache = {"prompt": "unidentified old output"}
        self.assertEqual(self.model().invoke_cached("prompt", cache), "model-a")
        self.assertEqual(cache["prompt"], "unidentified old output")

    def test_same_configuration_hits_and_different_prompt_misses(self):
        cache = {}
        model = self.model(seed=42)
        model.invoke_cached("prompt", cache)
        same = self.model(seed=42)
        same.invoke_cached("prompt", cache)
        same.invoke.assert_not_called()
        same.invoke_cached("other", cache)
        same.invoke.assert_called_once_with("other")

    def test_non_text_answer_cannot_enter_cache(self):
        model = self.model()
        model.invoke.return_value = None
        cache = {}
        with self.assertRaises(TypeError):
            model.invoke_cached("prompt", cache)
        self.assertEqual(cache, {})


if __name__ == "__main__":
    unittest.main()
