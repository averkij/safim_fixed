import unittest
from unittest.mock import Mock
from model_support import load_models, wrapper


class GenerationSeedTests(unittest.TestCase):
    def model(self, seed=0):
        module = load_models()
        model = wrapper(module, "StarcoderModel")
        model.seed = seed
        model.invoke = Mock(return_value="answer")
        return module, model

    def test_same_prompt_same_seed_independent_of_other_prompts(self):
        module, model = self.model(42)
        model.invoke_cached("a", {})
        first = module.transformers.set_seed.call_args
        self.assertIsNotNone(first)
        model.invoke_cached("b", {})
        model.invoke_cached("a", {})
        self.assertEqual(module.transformers.set_seed.call_args, first)

    def test_changed_seed_or_prompt_changes_rng_seed(self):
        seeds = []
        for base, prompt in [(1, "a"), (2, "a"), (1, "b")]:
            module, model = self.model(base)
            model.invoke_cached(prompt, {})
            self.assertIsNotNone(module.transformers.set_seed.call_args)
            seeds.append(module.transformers.set_seed.call_args.args[0])
        self.assertEqual(len(set(seeds)), 3)

    def test_base_seed_changes_cache_identity(self):
        cache = {}
        _, first = self.model(1)
        _, second = self.model(2)
        first.invoke_cached("a", cache)
        second.invoke_cached("a", cache)
        second.invoke.assert_called_once()

    def test_cache_hit_does_not_consume_rng(self):
        module, model = self.model()
        cache = {}
        model.invoke_cached("a", cache)
        self.assertEqual(module.transformers.set_seed.call_count, 1)
        model.invoke_cached("a", cache)
        self.assertEqual(module.transformers.set_seed.call_count, 1)


if __name__ == "__main__":
    unittest.main()
