from typing import List, Union, Set
from pyswip import Prolog
from lm_heuristic.prolog.utils import format_term, join
from lm_heuristic.prolog.parser import parse_to_prolog


class PrologGrammarEngine:
    """
    This class is used to :
        - initialize a interface toward Prolog using pyswip
        - transform and load a grammar into the prolog engine
        - define methods to query the grammar

    This class must be instanciate ony one
    """

    all_ready_initialize = False

    def __init__(self, path_to_methods_file: str):
        """
        :param path_to_methods_file: path toward prolog knowledge base that contains
                                     the predicates used to query the grammar
        """
        assert not PrologGrammarEngine.all_ready_initialize
        self.prolog = Prolog()
        self.prolog.consult(path_to_methods_file)

        # We keep in memory all the predicates that had been added to the prolog engine
        # in order to be able to remove them if needed
        self.current_predicates: List[str] = []

        # We keep in memoery all the terminal symbol of the grammar
        # in order to not have to communicate each time with prolog
        # when we need to know if a symbol is terminal
        self.terminals: Set[str] = set()

        PrologGrammarEngine.all_ready_initialize = True

    def delete_grammar(self):
        """
        Remove all the grammar predicates from the prolog engine
        """
        for rule in self.current_predicates:
            self.prolog.retractall(rule)
        self.current_predicates = []
        self.terminals = {}

    def retrieve_terminal(self):
        """
        Load the terminal symbol in memory once for all
        """
        answers = self.prolog.query("terminal(X)")
        self.terminals = {answer["X"] for answer in answers}

    def load_grammar(self, ntlk_str_grammar: str):
        """
        Transform the grammar into prolog predicates
        and load it in the prolog engine
        """
        self.current_predicates = parse_to_prolog(ntlk_str_grammar)
        for rule in self.current_predicates:
            self.prolog.assertz(rule)
        self.retrieve_terminal()

    def valid_children(self, symbols: List[str]) -> List[List[str]]:
        """
        Given a derivation (list of terms), return all the valide children node. Ie:
            - all symbols string that can derivate from this derivation using only one rule
            - all symbols string from which a terminal leaf can be reached
        """
        try:
            answer = next(self.prolog.query("all_valid_children([%s], X)" % join(symbols)))
            return format_term(answer["X"])
        except StopIteration:
            return []

    def leaf(self, symbols: List[str]) -> Union[List[str], None]:
        """
        Given a derivation, return a random terminal leaf if it exists, None else
        """
        answers = self.prolog.query("random_leaf([%s], X)" % join(symbols))
        try:
            answer = next(answers)
            return format_term(answer["X"])
        except StopIteration:
            return None

    def is_terminal(self, symbol: str) -> bool:
        """
        return true is the symbol is terminal
        """
        return symbol in self.terminals

    def set_random_seed(self, seed: int):
        """
        set the random seed of prolog engine
        """
        # WARNING - This does not seem to work !
        self.prolog.assertz("set_random(seed(%d))" % seed)
