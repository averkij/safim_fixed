import json
from pathlib import Path
import tempfile
import unittest
from parser_support import load_prompt
from source_support import source_at
from test_api_scoring import load_evaluator


class TaskVariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = load_prompt()

    def test_chat_modes_and_extraction_support_variants(self):
        for variant in ["block_v2", "control_fixed"]:
            for lang in ["python", "cpp", "java", "csharp"]:
                with self.subTest(variant=variant, lang=lang):
                    sample = dict(lang=lang, prompt=self.prompt.COMPLETION_PLACEHOLDER[lang], eval_prompt="{{completion}}")
                    for mode in ["instructed", "fewshot", "prefix_feeding"]:
                        self.assertIsInstance(self.prompt.apply_prompt(sample, variant, mode, None), str)
                    self.assertEqual(self.prompt.extract_code_from_chat_model("```\nx = 1\n```", lang, variant), "x = 1")

    def test_python_demonstration_colon_belongs_to_correct_side(self):
        self.assertTrue(self.prompt.FEW_SHOT_ANSWERS[("control", "python")].rstrip().endswith(":"))
        self.assertFalse(self.prompt.FEW_SHOT_ANSWERS[("control_fixed", "python")].rstrip().endswith(":"))
        self.assertIn("# TODO: Your code here:\n", self.prompt.FEW_SHOT_PROMPTS[("control_fixed", "python")])

    def test_colon_cleanup_requires_fixed_variant_and_suffix(self):
        sample = dict(lang="python", eval_prompt="if {{completion}}:\n    pass")
        self.assertEqual(self.prompt.apply_postprocessors("x:", sample, "control_fixed", ["truncate_control"]), "x")
        self.assertEqual(self.prompt.apply_postprocessors("x ", sample, "control_fixed", ["truncate_control"]), "x ")
        self.assertEqual(self.prompt.apply_postprocessors("x :", sample, "control_fixed", ["truncate_control"]), "x ")
        self.assertEqual(self.prompt.apply_postprocessors("x:", sample, "control", ["truncate_control"]), "x:")
        sample["eval_prompt"] = "if {{completion}}\n    pass"
        self.assertEqual(self.prompt.apply_postprocessors("x:", sample, "control_fixed", ["truncate_control"]), "x:")

    def test_evaluation_rejects_declared_variant_mismatch(self):
        module = load_evaluator(source_at("evaluate.py"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.jsonl"
            Path(str(path) + ".run.json").write_text(json.dumps(dict(configuration=dict(completion_type="control"))))
            with self.assertRaises(ValueError):
                module.validate_input_variant(path, "control_fixed")
            module.validate_input_variant(path, "control")


if __name__ == "__main__":
    unittest.main()
