from argparse import Namespace
import unittest
from unittest.mock import Mock, patch
from model_support import load_models, wrapper


class GenerationBudgetTests(unittest.TestCase):
    def test_pipeline_wrappers_honor_requested_budget(self):
        module = load_models()
        for name in ["CodeLlama", "MagicCoderModel"]:
            with self.subTest(name=name):
                model = wrapper(module, name)
                model.max_new_tokens = 256
                model.invoke("prompt")
                self.assertEqual(model.pipeline.call_args.kwargs["max_new_tokens"], 256)

    def test_direct_wrappers_honor_requested_budget(self):
        module = load_models()
        for name in ["Incoder", "CodegenModel", "StarcoderModel", "DeepseekModel", "PhiModel", "MixtralModel", "WizardModel", "SantacoderModel"]:
            with self.subTest(name=name):
                model = wrapper(module, name)
                model.max_new_tokens = 256
                model.invoke("prompt")
                kwargs = model.model.generate.call_args.kwargs
                budget = kwargs["max_new_tokens"] if "max_new_tokens" in kwargs else kwargs["max_length"] - 10
                self.assertEqual(budget, 256)

    def test_openai_wrapper_has_explicit_cap(self):
        module = load_models()
        model = wrapper(module, "OpenAIModel")
        model.max_new_tokens = 256
        model.invoke("prompt")
        self.assertEqual(module.openai.ChatCompletion.create.call_args.kwargs["max_tokens"], 256)

    def test_builder_retains_default_and_applies_override(self):
        module = load_models()
        for budget in [128, 512]:
            args = Namespace(model_name="bigcode/starcoder-fixture", block_comments=False, max_new_tokens=budget)
            with patch.object(module, "StarcoderModel", return_value=Mock()) as constructor:
                model = module.build_model(args)
                self.assertEqual(model.max_new_tokens, budget)
                constructor.assert_called_once()

    def test_invalid_budget_rejected_before_model_loading(self):
        module = load_models()
        with patch.object(module, "StarcoderModel") as constructor:
            with self.assertRaises(ValueError):
                module.build_model(Namespace(model_name="bigcode/starcoder-fixture", block_comments=False, max_new_tokens=0))
            constructor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
