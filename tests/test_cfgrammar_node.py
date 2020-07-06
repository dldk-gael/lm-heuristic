import pytest
import nltk

from lm_heuristic.tree.interface.nltk_grammar import CFGrammarNode

# pylint: disable=redefined-outer-name, missing-function-docstring

@pytest.fixture
def toy_nltk_grammar():
    grammar_str = """
    s -> n v
    n -> 'gael'
    v -> 'drink'"""
    return nltk.CFG.fromstring(grammar_str)


@pytest.fixture
def root(toy_nltk_grammar):
    return CFGrammarNode(toy_nltk_grammar.start(), toy_nltk_grammar)


def test_str_representation(root):
    assert str(root) == "s."


def test_children(root):
    children = root.children()
    assert len(root.children()) == 1
    assert str(children[0]) == "n v."


def test_shrink_option(root):
    root.shrink = True
    assert len(root.children()) == 2


def test_terminal(root):
    assert root.is_terminal() is False
    node = root
    for _ in range(3):
        node = node.random_children()
    assert node.is_terminal()
