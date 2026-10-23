"""Python API comparisons must preserve literal content and token boundaries."""

import unittest

from test_api_scoring import EVALUATE_SOURCE, load_evaluator


class ApiWhitespaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator(EVALUATE_SOURCE)

    def match(self, candidate, reference):
        return self.evaluator.syntax_match(candidate, reference, "python")

    def test_string_spaces(self):
        self.assertFalse(self.match('f("a b")', 'f("ab")'))

    def test_string_tab(self):
        self.assertFalse(self.match('f("a\tb")', 'f("ab")'))

    def test_multiline_string(self):
        self.assertFalse(self.match('f("""a\nb""")', 'f("""ab""")'))

    def test_unicode_space(self):
        self.assertFalse(self.match('f("a\u00a0b")', 'f("ab")'))

    def test_bytes_literal(self):
        self.assertFalse(self.match('f(b"a b")', 'f(b"ab")'))

    def test_raw_string(self):
        self.assertFalse(self.match('f(r"a b")', 'f(r"ab")'))

    def test_fstring_literal_part(self):
        self.assertFalse(self.match('f(f"a b{x}")', 'f(f"ab{x}")'))

    def test_nested_and_keyword_literals(self):
        self.assertFalse(self.match('f({"key": ["a b"]})', 'f({"key": ["ab"]})'))
        self.assertFalse(self.match('f(label="a b")', 'f(label="ab")'))

    def test_non_call_literal(self):
        self.assertFalse(self.match('"a b"', '"ab"'))

    def test_token_boundaries(self):
        self.assertFalse(self.match('f(a is not b)', 'f(aisnotb)'))
        self.assertFalse(self.match('f(a b)', 'f(ab)'))

    def test_meaningful_whitespace_preserved_with_formatting(self):
        self.assertTrue(self.match('  f( "a b",\n x = 2 )  ', 'f("a b",x=2)'))
        self.assertTrue(self.match('f("a\\tb")', 'f("a\tb")'))
        self.assertTrue(self.match("f('a b')", 'f("a b")'))

    def test_comprehension_formatting(self):
        self.assertTrue(self.match('f([i for i in range(3)])', 'f([ i for i in range(3) ])'))
        self.assertFalse(self.match('f([i for i in range(3)])', 'f([iforiinrange(3)])'))

    def test_invalid_expression_fallback(self):
        self.assertTrue(self.match(' f( ', 'f('))
        self.assertFalse(self.match('f("a b",', 'f("ab",'))
        self.assertFalse(self.match('f( x', 'f(x'))


if __name__ == "__main__":
    unittest.main()
