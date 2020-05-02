import pytest
import nltk
from src.tree_search.tree import Derivation
from src.tree_search.strategy import RandomSearch


@pytest.fixture
def random_searcher():
    grammar_str = """
    s -> n v
    n -> 'gael'
    v -> 'eats'"""
    grammar = nltk.CFG.fromstring(grammar_str)
    root = Derivation(grammar.start(), grammar)
    return RandomSearch(
        root=root,
        evaluation_fn=lambda nodes: [0] * len(nodes),
        n_samples=1,
        batch_size=1,
    )


def test_search(random_searcher):
    assert str(random_searcher()) == "gael eats."


def test_path(random_searcher):
    with pytest.raises(AssertionError):
        random_searcher.path()

    random_searcher.search()
    path = random_searcher.path()
    assert str(path[0]) == "s."
    assert str(path[-1]) == "gael eats."
    assert len(path) == 4
