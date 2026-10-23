"""API matching must not discard repeated keyword or unpacking arguments."""

import unittest

from test_api_scoring import EVALUATE_SOURCE, load_evaluator


class ApiKeywordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = load_evaluator(EVALUATE_SOURCE)

    def match(self, candidate, reference):
        return self.evaluator.syntax_match(candidate, reference, "python")

    def test_changed_first_unpacking(self):
        self.assertFalse(self.match('f(**a, **b)', 'f(**c, **b)'))

    def test_omitted_unpacking(self):
        self.assertFalse(self.match('f(**a, **b)', 'f(**b)'))

    def test_repeated_identical_unpacking(self):
        self.assertFalse(self.match('f(**a, **a)', 'f(**a)'))

    def test_swapped_earlier_unpackings(self):
        self.assertFalse(self.match('f(**a, **b, **c)', 'f(**b, **a, **c)'))

    def test_named_keyword_interleaving(self):
        self.assertFalse(self.match('f(x=1, **a)', 'f(**a, x=1)'))

    def test_duplicate_named_keyword(self):
        self.assertFalse(self.match('f(axis=-1, axis=0)', 'f(axis=0)'))

    def test_duplicate_identical_named_keyword(self):
        self.assertFalse(self.match('f(axis=0, axis=0)', 'f(axis=0)'))

    def test_duplicate_keyword_cannot_hide_different_value(self):
        self.assertFalse(self.match('f(axis=1, axis=0)', 'f(axis=2, axis=0)'))

    def test_identical_unpackings(self):
        self.assertTrue(self.match('f(**a, **b)', 'f(**a, **b)'))
        self.assertTrue(self.match('f(x=1, **a, y=2, **b)', 'f(x=1, **a, y=2, **b)'))

    def test_named_keyword_order_preserved(self):
        self.assertTrue(self.match('f(a=1, b=2)', 'f(b=2, a=1)'))
        self.assertTrue(self.match('f(*xs, a=1, b=2)', 'f(*xs, b=2, a=1)'))

    def test_single_unpacking_control(self):
        self.assertTrue(self.match('f(**a)', 'f( **a )'))
        self.assertFalse(self.match('f(**a)', 'f(**b)'))
        self.assertFalse(self.match('f(**a)', 'f()'))

    def test_positional_and_callee_controls(self):
        self.assertFalse(self.match('f(1, **a)', 'f(2, **a)'))
        self.assertFalse(self.match('f(**a)', 'g(**a)'))


if __name__ == "__main__":
    unittest.main()
