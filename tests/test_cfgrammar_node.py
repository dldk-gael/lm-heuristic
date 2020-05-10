import pytest
import nltk

from lm_heuristic.tree import CFGrammarNode

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


def test_childrens(root):
    childrens = root.childrens()
    assert len(root.childrens()) == 1
    assert str(childrens[0]) == "n v."


def test_shrink_option(root):
    root.shrink = True
    assert len(root.childrens()) == 2


def test_terminal(root):
    assert root.is_terminal() is False
    node = root
    for _ in range(3):
        node = node.random_children()
    assert node.is_terminal()
