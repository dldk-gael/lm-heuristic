import pytest
import nltk
from src.tree_search.tree import Derivation


@pytest.fixture
def toy_grammar():
    grammar_str = """
    s -> n v
    n -> 'gael'
    v -> 'drink'"""
    return nltk.CFG.fromstring(grammar_str)


def test_init_grammar_root(toy_grammar):
    Derivation(toy_grammar.start(), toy_grammar)


def test_str_representation(toy_grammar):
    assert str(Derivation(toy_grammar.start(), toy_grammar)) == "s."


def test_childrens(toy_grammar):
    root = Derivation(toy_grammar.start(), toy_grammar)
    childrens = root.childrens()
    assert len(root.childrens()) == 1
    assert str(childrens[0]) == "n v."


def test_shrink_option(toy_grammar):
    root = Derivation(toy_grammar.start(), toy_grammar, shrink=True)
    assert len(root.childrens()) == 2


def test_root_is_not_terminal(toy_grammar):
    root = Derivation(toy_grammar.start(), toy_grammar)
    assert root.is_terminal() is False


def test_leaf_is_terminal(toy_grammar):
    root = Derivation(toy_grammar.start(), toy_grammar)
    node = root
    for i in range(3):
        node = node.random_children()
    assert node.is_terminal()
