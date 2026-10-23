import unittest
from parser_support import load_prompt


class MacroCompletionTests(unittest.TestCase):
    def test_defined_statement_macro_inside_hole_is_preserved(self):
        module = load_prompt()
        sample = dict(lang="cpp", eval_prompt="#define LOOP(i,s,n) for(int i=s;i<n;i++)\nint main() { {{completion}} }")
        completion = "int n=2;\nLOOP(i,0,n) n += i;"
        for name in ["truncate_line_until_parsable", "truncate_line_until_block"]:
            with self.subTest(processor=name):
                self.assertEqual(getattr(module, name)(sample, completion), completion)

    def test_unused_macro_does_not_disable_normal_check(self):
        module = load_prompt()
        sample = dict(lang="cpp", eval_prompt="#define LOOP(i,n) for(int i=0;i<n;i++)\nint main() { {{completion}} }")
        self.assertEqual(module.truncate_line_until_parsable(sample, "return 1"), "")

    def test_ordinary_function_is_not_a_macro(self):
        module = load_prompt()
        sample = dict(lang="cpp", eval_prompt="int f(int x) { return x; }\nint main() { {{completion}} }")
        self.assertEqual(module.truncate_line_until_parsable(sample, "f(1)"), "")


if __name__ == "__main__":
    unittest.main()
