import unittest
from parser_support import load_ast, load_prompt


class MissingSyntaxTests(unittest.TestCase):
    def test_recovered_missing_tokens_are_errors(self):
        module = load_ast()
        for lang, code in [("java", "class A { int x = 1 }"), ("cpp", "int main() { return 1 }"),
                           ("csharp", "class A { void F() { F(1; } }")]:
            with self.subTest(lang=lang):
                tree = module.get_parser(lang).parse(code.encode())
                self.assertTrue(tree.root_node.has_error)
                visitor = module.ErrorCheckVisitor()
                visitor(tree)
                self.assertGreater(visitor.error_cnt, 0)

    def test_valid_and_explicit_error_controls(self):
        module = load_ast()
        for code, expected in [("x = 1\n", 0), ("if True\n    pass\n", 1)]:
            visitor = module.ErrorCheckVisitor()
            visitor(module.get_parser("python").parse(code.encode()))
            self.assertEqual(visitor.error_cnt, expected)

    def test_parsable_postprocessor_does_not_keep_missing_token(self):
        module = load_prompt()
        sample = dict(lang="cpp", eval_prompt="int main() {\n{{completion}}\n}")
        self.assertEqual(module.truncate_line_until_parsable(sample, "return 1"), "")
        self.assertEqual(module.truncate_line_until_parsable(sample, "return 1;"), "return 1;")


if __name__ == "__main__":
    unittest.main()
