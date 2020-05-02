import pytest
import nltk
from src.tree_search.tree import Derivation
from src.tree_search.strategy import MonteCarloTreeSearch


@pytest.fixture
def toy_grammar():
    grammar_str = """
        s -> snp vp
        vp -> v onp
        onp -> np
        onp -> obj
        snp -> np
        snp -> subj
        subj -> 'Bas'
        subj -> 'he'
        obj -> 'Piet'
        obj -> 'him'
        np -> det n
        det -> 'the'
        n -> 'man'
        v -> 'knows'
        """
    return nltk.CFG.fromstring(grammar_str)


@pytest.fixture
def basic_mcts(toy_grammar):
    root = Derivation(toy_grammar.start(), toy_grammar)
    return MonteCarloTreeSearch(
        root=root, evaluation_fn=lambda nodes: [0] * len(nodes), nb_of_tree_walks=1
    )


def test_mcts_search_find_a_leaf(basic_mcts):
    assert basic_mcts.search().is_terminal()


def test_mcts_path_raise_assertion_error_when_no_search(basic_mcts):
    with pytest.raises(AssertionError):
        basic_mcts.path()


def test_mcts_path(basic_mcts):
    path = basic_mcts.path()
    print(path)
    assert str(path[0]) == "s."
    assert path[-1].is_terminal()
