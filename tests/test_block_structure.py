import unittest
from parser_support import load_prompt


class BlockStructureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_prompt()

    def test_pass_reference_survives(self):
        sample = dict(lang="python", eval_prompt="if ready:\n    {{completion}}\nprint('tail')\n")
        self.assertEqual(self.module.truncate_line_until_block(sample, "pass"), "pass")

    def test_nested_python_completion_preserves_suffix(self):
        sample = dict(lang="python", eval_prompt="def f():\n    {{completion}}\n    return y\n")
        code = "if x:\n        y = 1\n    else:\n        y = 2"
        self.assertEqual(self.module.truncate_line_until_block(sample, code), code)

    def test_new_if_cannot_capture_existing_else(self):
        sample = dict(lang="python", eval_prompt="if x:\n    {{completion}}\nelse:\n    fallback()\n")
        code = "work()\nif other:\n    work()"
        self.assertEqual(self.module.truncate_line_until_block(sample, code), "work()\n")

    def test_python_suffix_statement_keeps_its_parent(self):
        sample = dict(lang="python", eval_prompt="def f():\n    {{completion}}\n    return 1\n")
        self.assertEqual(self.module.truncate_line_until_block(sample, "pass\n    # extra"), "pass\n    # extra")

    def test_c_family_nested_block_controls(self):
        for lang, template in [("cpp", "int main() { {{completion}} return 0; }"),
                               ("java", "class A { void f() { {{completion}} int y=2; } }"),
                               ("csharp", "class A { void F() { {{completion}} int y=2; } }")]:
            with self.subTest(lang=lang):
                code = "if (true) { int x=1; }"
                self.assertEqual(self.module.truncate_line_until_block(dict(lang=lang, eval_prompt=template), code), code)


if __name__ == "__main__":
    unittest.main()
