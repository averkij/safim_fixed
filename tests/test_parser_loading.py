import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from parser_support import load_ast


class ParserLoadingTests(unittest.TestCase):
    def test_import_does_not_build_or_require_grammars(self):
        import sys
        with patch.dict(sys.modules, {"tree_sitter": None}):
            self.assertTrue(callable(load_ast().get_parser))

    def test_four_languages_parse_from_unrelated_working_directory(self):
        examples = {"python": b"x = 1\n", "java": b"class A { int x = 1; }", "cpp": b"int x = 1;", "csharp": b"class A { int x = 1; }"}
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"SAFIM_PARSER_BACKEND": "wheels"}):
            try:
                os.chdir(directory)
                module = load_ast()
                for lang, text in examples.items():
                    with self.subTest(lang=lang):
                        tree = module.get_parser(lang).parse(text)
                        self.assertFalse(tree.root_node.has_error)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
