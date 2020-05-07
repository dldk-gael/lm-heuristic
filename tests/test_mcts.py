import pytest
import nltk

from lm_heuristic.tree import CFGrammarNode
from lm_heuristic.tree_search.mcts import MonteCarloTreeSearch
from lm_heuristic.heuristic import Heuristic


@pytest.fixture
def root():
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
    toy_grammar = nltk.CFG.fromstring(grammar_str)
    return CFGrammarNode(toy_grammar.start(), toy_grammar)


@pytest.fixture
def basic_mcts():
    heuristic = Heuristic(lambda nodes: [0] * len(nodes))
    return MonteCarloTreeSearch(heuristic)


def test_mcts_search_find_a_leaf(basic_mcts, root):
    best_leaf, value = basic_mcts(root, nb_of_tree_walks=1)
    assert best_leaf.is_terminal()


def test_mcts_path_raise_assertion_error_when_no_search(basic_mcts):
    with pytest.raises(AssertionError):
        basic_mcts.path()


def test_mcts_path(basic_mcts, root):
    basic_mcts(root, nb_of_tree_walks=1)
    path = basic_mcts.path()
    assert str(path[0]) == "s."
    assert path[-1].is_terminal()
