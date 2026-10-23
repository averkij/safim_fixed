import unittest
from parser_support import load_prompt


class CodeFenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_prompt()

    def extract(self, text):
        return self.module.extract_code_from_chat_model(text, "python", "block_v2")

    def test_indented_fence_preserves_relative_indentation(self):
        self.assertEqual(self.extract("Answer:\n  ```python\n  if x:\n      y()\n  ```"), "if x:\n    y()")

    def test_unclosed_fence_retains_generated_code(self):
        self.assertEqual(self.extract("```python\nx = 1\ny = 2"), "x = 1\ny = 2")

    def test_long_fence_does_not_close_on_short_fence(self):
        self.assertEqual(self.extract('````python\ns = """\n```\n"""\n````'), 's = """\n```\n"""')

    def test_tilde_fence(self):
        self.assertEqual(self.extract("~~~python\nx = 1\n~~~"), "x = 1")

    def test_plain_inline_and_standard_controls(self):
        self.assertEqual(self.extract("x = 1"), "x = 1")
        self.assertEqual(self.extract("Use `x = 1`."), "x = 1")
        self.assertEqual(self.extract("```python\nx = 1\n```"), "x = 1")
        self.assertEqual(self.extract("```python"), "")


if __name__ == "__main__":
    unittest.main()
