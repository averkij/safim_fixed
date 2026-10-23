import unittest
from parser_support import load_prompt


class UnparsableContextTests(unittest.TestCase):
    def test_valid_macro_context_does_not_erase_candidate(self):
        module = load_prompt()
        sample = dict(lang="cpp", eval_prompt="#define LOOP(i,n) for(int i=0;i<n;i++)\nint main() { LOOP(i,2) { {{completion}} } }")
        for name in ["truncate_line_until_parsable", "truncate_line_until_block"]:
            with self.subTest(processor=name):
                self.assertEqual(getattr(module, name)(sample, "int x = i;"), "int x = i;")

    def test_normal_context_still_checks_candidate(self):
        module = load_prompt()
        sample = dict(lang="cpp", eval_prompt="int main() { {{completion}} }")
        self.assertEqual(module.truncate_line_until_parsable(sample, "return 1;"), "return 1;")
        self.assertEqual(module.truncate_line_until_parsable(sample, "return 1"), "")

    def test_fallback_does_not_invent_a_pass_or_repair_code(self):
        module = load_prompt()
        sample = dict(lang="cpp", eval_prompt="#define LOOP(i,n) for(int i=0;i<n;i++)\nint main() { LOOP(i,2) { {{completion}} } }")
        self.assertEqual(module.truncate_line_until_parsable(sample, "invalid generated text !!!"), "invalid generated text !!!")


if __name__ == "__main__":
    unittest.main()
