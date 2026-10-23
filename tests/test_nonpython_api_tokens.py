import unittest
from source_support import source_at
from test_api_scoring import load_evaluator


class NonPythonApiTokenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_evaluator(source_at("evaluate.py"))

    def test_string_whitespace_is_semantic(self):
        for lang in ["cpp", "java", "csharp"]:
            for reference, candidate in [('f("a b")', 'f("ab")'), ('f("a  b")', 'f("a b")'), ('f("é b")', 'f("éb")')]:
                with self.subTest(lang=lang, reference=reference):
                    self.assertFalse(self.module.syntax_match(candidate, reference, lang))

    def test_token_boundaries_cannot_be_merged(self):
        for lang in ["cpp", "java", "csharp"]:
            with self.subTest(lang=lang):
                self.assertFalse(self.module.syntax_match("f(a + +b)", "f(a++b)", lang))

    def test_formatting_and_comments_are_ignored(self):
        for lang in ["cpp", "java", "csharp"]:
            with self.subTest(lang=lang):
                self.assertTrue(self.module.syntax_match('f ( "a b" , x )', 'f("a b",x)', lang))
                self.assertTrue(self.module.syntax_match('f(/* note */ x)', 'f(x)', lang))

    def test_raw_and_verbatim_literals(self):
        self.assertFalse(self.module.syntax_match('f(R"(a b)")', 'f(R"(ab)")', "cpp"))
        self.assertFalse(self.module.syntax_match('F(@"a b")', 'F(@"ab")', "csharp"))

    def test_cpp_declaration_reference_is_supported(self):
        self.assertTrue(self.module.syntax_match('boost::filesystem::path dir( "example_dir")', 'boost::filesystem::path dir("example_dir")', "cpp"))
        self.assertFalse(self.module.syntax_match('boost::filesystem::path dir("example dir")', 'boost::filesystem::path dir("exampledir")', "cpp"))

    def test_identical_invalid_fallback(self):
        for lang in ["cpp", "java", "csharp"]:
            with self.subTest(lang=lang):
                self.assertTrue(self.module.syntax_match('f("unterminated', 'f("unterminated', lang))
                self.assertFalse(self.module.syntax_match('f("a b', 'f("ab', lang))


if __name__ == "__main__":
    unittest.main()
