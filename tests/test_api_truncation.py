import unittest
from parser_support import load_prompt


class ApiTruncationTests(unittest.TestCase):
    def truncate(self, code, lang):
        return load_prompt().apply_postprocessors(code, dict(lang=lang), "api", ["truncate_api_call"])

    def test_call_chains_preserved(self):
        for lang, code in [("python", "a(x).b()"), ("java", "new Builder().build()"),
                           ("cpp", "a(x).b()"), ("csharp", "a(x).B()")]:
            with self.subTest(lang=lang):
                self.assertEqual(self.truncate(code, lang), code)

    def test_quoted_parentheses_and_unicode_preserved(self):
        for lang in ["python", "java", "cpp", "csharp"]:
            with self.subTest(lang=lang):
                code = 'f("é )(").g()'
                self.assertEqual(self.truncate(code, lang), code)

    def test_subscript_and_nested_calls(self):
        self.assertEqual(self.truncate('f(g(1))[0].h()', "python"), 'f(g(1))[0].h()')

    def test_trailing_statement_removed(self):
        for lang, code in [("python", "f().g()\nother()"), ("cpp", "f().g(); other();"),
                           ("java", "f().g(); other();"), ("csharp", "f().g(); other();")]:
            with self.subTest(lang=lang):
                self.assertEqual(self.truncate(code, lang), "f().g()")

    def test_simple_call_and_unrecognizable_text_controls(self):
        self.assertEqual(self.truncate("f(x)", "python"), "f(x)")
        self.assertEqual(self.truncate("unfinished text", "python"), "unfinished text")


if __name__ == "__main__":
    unittest.main()
