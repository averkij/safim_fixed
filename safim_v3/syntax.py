"""Byte-exact masks for the four supported Tree-sitter grammars."""
import importlib
from functools import lru_cache

from tree_sitter import Language, Parser


@lru_cache(maxsize=4)
def parser(lang):
    module = importlib.import_module("tree_sitter_" + {"csharp": "c_sharp"}.get(lang, lang))
    return Parser(Language(module.language()))


def walk(node, depth=0):
    yield node, depth
    for child in node.named_children:
        yield from walk(child, depth + 1)


def parse_tree(source, lang):
    # Some grammars require a newline after an EOF preprocessor directive,
    # although the compiler accepts EOF itself. A virtual trailing newline does
    # not move any original byte offset or alter the stored/reconstructed source.
    code = source.encode("utf-8")
    return parser(lang).parse(code if code.endswith(b"\n") else code + b"\n")


def masks(source, lang):
    code = source.encode("utf-8")
    root = parse_tree(source, lang).root_node
    if root.has_error:
        raise ValueError("Original source does not parse")
    candidates = []
    for node, depth in walk(root):
        if node.type in ("block", "compound_statement"):
            children = [c for c in node.named_children if c.type not in ("comment",)]
            if not children:
                continue
            start, end = children[0].start_byte, children[-1].end_byte
            candidates.append({"task_type": "block", "start_byte": start, "end_byte": end,
                               "node_type": node.type, "depth": depth,
                               "statement_count": len(children),
                               "mutations": ["pass" if lang == "python" else ";"]})
        if node.type in ("if_statement", "elif_clause", "while_statement", "do_statement", "for_statement",
                         "for_in_statement", "for_range_loop", "enhanced_for_statement", "foreach_statement"):
            condition = node.child_by_field_name("condition")
            field = "condition"
            if condition is None:
                field = "right" if lang in ("python", "csharp", "cpp") else "value"
                condition = node.child_by_field_name(field)
            if condition is None:
                continue
            if condition.type == "condition_clause":
                condition = condition.child_by_field_name("value")
                if condition is None:
                    continue
            if field == "condition":
                alternatives = ["True", "False"] if lang == "python" else ["true", "false"]
                if lang == "java":
                    # Literal loop conditions trigger javac's compile-time
                    # unreachable-statement checks. These preserve true/false
                    # semantics without being Java constant expressions.
                    alternatives = ["java.lang.Boolean.TRUE.booleanValue()",
                                    "java.lang.Boolean.FALSE.booleanValue()"]
                if condition.type == "parenthesized_expression":
                    alternatives = ["(" + a + ")" for a in alternatives]
            elif lang == "python":
                alternatives = ["[]"]
            else:
                # Typed empty iterables require type information; no guessed replacements.
                continue
            candidates.append({"task_type": "control", "start_byte": condition.start_byte,
                "end_byte": condition.end_byte, "node_type": condition.type, "depth": depth,
                "mutations": alternatives})
    return sorted(candidates, key=lambda m: (-m["depth"], m["start_byte"], m["end_byte"]))


def replace(source, mask, completion):
    raw = source.encode("utf-8")
    return (raw[:mask["start_byte"]] + completion.encode("utf-8") + raw[mask["end_byte"]:]).decode("utf-8")


def syntax_valid(source, lang):
    return not parse_tree(source, lang).root_node.has_error
