import json
import unittest
from parser_support import load_prompt


class ApiDemonstrationTests(unittest.TestCase):
    def test_completed_examples_parse_in_each_language(self):
        prompt=load_prompt()
        for lang in ["python","java","cpp","csharp"]:
            with self.subTest(lang=lang):
                code=prompt.FEW_SHOT_PROMPTS[("api",lang)].replace(
                    prompt.COMPLETION_PLACEHOLDER[lang],prompt.FEW_SHOT_ANSWERS[("api",lang)].rstrip())
                if code.startswith("Complete the code"):
                    code=code.split("\n\n",1)[1]
                tree=prompt.get_parser(lang).parse(code.encode())
                print(json.dumps(dict(lang=lang,code=code,has_error=tree.root_node.has_error)))
                self.assertFalse(tree.root_node.has_error)


if __name__=="__main__":unittest.main()
