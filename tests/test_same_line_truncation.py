import unittest
from parser_support import load_prompt


class SameLineTruncationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_prompt()

    def test_excess_closing_brace_keeps_valid_statement(self):
        for lang, template in [("cpp", "int main() { {{completion}} }"),
                               ("java", "class A { void f() { {{completion}} } }"),
                               ("csharp", "class A { void F() { {{completion}} } }")]:
            for processor in ["truncate_line_until_block", "truncate_line_until_parsable"]:
                with self.subTest(lang=lang, processor=processor):
                    actual = getattr(self.module, processor)(dict(lang=lang, eval_prompt=template), "int x=1; }")
                    self.assertEqual(actual.rstrip(), "int x=1;")

    def test_literal_brace_and_unicode_are_not_split(self):
        sample = dict(lang="cpp", eval_prompt="int main() { {{completion}} }")
        code = 'const char *s="é}"; }'
        self.assertEqual(self.module.truncate_line_until_block(sample, code).rstrip(), 'const char *s="é}";')

    def test_valid_nested_block_and_empty_controls(self):
        sample = dict(lang="cpp", eval_prompt="int main() { {{completion}} }")
        code = "if (true) { int x=1; }"
        self.assertEqual(self.module.truncate_line_until_block(sample, code), code)
        self.assertEqual(self.module.truncate_line_until_block(sample, ""), "")


if __name__ == "__main__":
    unittest.main()
