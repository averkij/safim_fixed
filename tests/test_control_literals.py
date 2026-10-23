import unittest
from parser_support import load_prompt


class ControlLiteralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_prompt()

    def test_strings_and_characters_do_not_end_condition(self):
        for lang in ["cpp", "java", "csharp"]:
            for condition in ['s == "é)"', "c == ')'", 'f("(") && g(")")', 's == "\\\")"']:
                with self.subTest(lang=lang, condition=condition):
                    self.assertEqual(self.module.truncate_control(dict(lang=lang), condition + ") { body; }"), condition)

    def test_comments_do_not_change_depth(self):
        for lang in ["cpp", "java", "csharp"]:
            with self.subTest(lang=lang):
                code = "x /* ) ( */ && f(1)"
                self.assertEqual(self.module.truncate_control(dict(lang=lang), code + ") {}"), code)

    def test_cpp_raw_and_csharp_verbatim_strings(self):
        for lang, code in [("cpp", 's == R"tag(()))tag"'), ("csharp", 's == @"a""b)"')]:
            with self.subTest(lang=lang):
                self.assertEqual(self.module.truncate_control(dict(lang=lang), code + ") {}"), code)

    def test_nested_and_python_controls(self):
        self.assertEqual(self.module.truncate_control(dict(lang="cpp"), "f(g(x)) && y) {}"), "f(g(x)) && y")
        self.assertEqual(self.module.truncate_control(dict(lang="python"), 'x == ")"\nextra'), 'x == ")"')


if __name__ == "__main__":
    unittest.main()
