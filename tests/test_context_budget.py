import unittest
from model_support import load_models, wrapper


class ContextBudgetTests(unittest.TestCase):
    def test_long_fim_preserves_structure_near_hole_and_budget(self):
        module = load_models()
        for name in ["StarcoderModel", "CodeLlama", "DeepseekModel", "SantacoderModel", "Incoder"]:
            for reverse in [False, True]:
                with self.subTest(name=name, reverse=reverse):
                    model = wrapper(module, name)
                    model.max_length, model.max_new_tokens = 160, 20
                    prompt = model.fit_infilling_prompt("a" * 300 + "LEFT", "RIGHT" + "b" * 300, reverse)
                    self.assertLessEqual(len(prompt) + 20, 160)
                    self.assertIn("LEFT", prompt)
                    self.assertIn("RIGHT", prompt)
                    # Compare marker sequence with an empty assembled prompt.
                    marker_text = prompt.replace("a", "").replace("b", "").replace("LEFT", "").replace("RIGHT", "")
                    expected = model.assemble_infilling_prompt("", "", reverse).replace("a", "").replace("b", "")
                    self.assertEqual(marker_text, expected)

    def test_short_prompt_unchanged(self):
        model = wrapper(load_models(), "StarcoderModel")
        self.assertEqual(model.fit_infilling_prompt("left", "right"), model.assemble_infilling_prompt("left", "right"))

    def test_impossible_budget_raises(self):
        model = wrapper(load_models(), "StarcoderModel")
        model.max_length = 128
        with self.assertRaises(ValueError):
            model.fit_infilling_prompt("left", "right")

    def test_direct_wrappers_reject_instead_of_silently_clipping(self):
        module = load_models()
        for name in ["Incoder", "CodegenModel", "StarcoderModel", "DeepseekModel", "PhiModel", "MixtralModel", "WizardModel", "SantacoderModel"]:
            with self.subTest(name=name):
                model = wrapper(module, name)
                model.max_length = 137  # Fixture input is 10 tokens, output 128.
                with self.assertRaises(ValueError):
                    model.invoke("prompt")
                model.model.generate.assert_not_called()

    def test_pipeline_and_remote_reject_unstructured_overflow(self):
        for name in ["CodeLlama", "MagicCoderModel", "OpenAIModel"]:
            with self.subTest(name=name):
                model = wrapper(load_models(), name)
                model.max_length = 140
                with self.assertRaises(ValueError):
                    model.invoke("x" * 200)


if __name__ == "__main__":
    unittest.main()
