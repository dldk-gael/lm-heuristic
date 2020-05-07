import pytest
import nltk
from tree import CFGrammarNode
from lm_heuristic.tree_search.random import RandomSearch
from lm_heuristic.heuristic import Heuristic


@pytest.fixture
def toy_grammar():
    grammar_str = """
    s -> n v
    n -> 'gael'
    v -> 'eats'"""
    grammar = nltk.CFG.fromstring(grammar_str)
    return CFGrammarNode(grammar.start(), grammar)


@pytest.fixture
def basic_random_searcher():
    return RandomSearch(heuristic=Heuristic(lambda nodes: [0] * len(nodes)),)


def test_search(toy_grammar, basic_random_searcher):
    best_leaf, _ = basic_random_searcher(toy_grammar, nb_of_tree_walks=1)
    assert str(best_leaf) == "gael eats."


def test_path(toy_grammar, basic_random_searcher):
    with pytest.raises(AssertionError):
        basic_random_searcher.path()

    _ = basic_random_searcher(toy_grammar, nb_of_tree_walks=10)
    path = basic_random_searcher.path()
    assert str(path[0]) == "s."
    assert str(path[-1]) == "gael eats."
    assert len(path) == 4
