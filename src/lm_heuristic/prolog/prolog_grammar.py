from typing import List, Union
from pyswip import Prolog
from lm_heuristic.prolog.utils import convert_grammar_to_prolog, format_term, join


class PrologGrammarEngine:
    all_ready_initialize = False

    def __init__(self, path_to_methods_file):
        """
        Must be initialize only once.
        """
        assert not PrologGrammarEngine.all_ready_initialize
        self.prolog = Prolog()
        self.prolog.consult(path_to_methods_file)

        self.current_predicates = []
        self.terminals = {}
        PrologGrammarEngine.all_ready_initialize = True

        self.valid_children_counter = 0
        self.valid_children_with_rleaf_counter = 0
        self.leaf_counter = 0

    def delete_grammar(self):
        for rule in self.current_predicates:
            self.prolog.retractall(rule)
        self.current_predicates = []
        self.terminals = {}

    def retrieve_terminal(self):
        answers = self.prolog.query("terminal(X)")
        self.terminals = {answer["X"] for answer in answers}

    def load_grammar(self, ntlk_str_grammar: str):
        self.current_predicates = convert_grammar_to_prolog(ntlk_str_grammar)
        for rule in self.current_predicates:
            self.prolog.assertz(rule)
            # print("%s." % rule)
        self.retrieve_terminal()

    def valid_children(self, symbols: List[str]) -> List[List[str]]:
        self.valid_children_counter += 1
        try:
            answer = next(self.prolog.query("all_valid_children([%s], X)" % join(symbols)))
            return format_term(answer["X"])
        except StopIteration:
            return []

    def leaf(self, symbols: List[str]) -> Union[List[str], None]:
        self.leaf_counter += 1
        answers = self.prolog.query("random_leaf([%s], X)" % join(symbols))
        # TODO : add random here. How ?!
        try:
            answer = next(answers)
            return format_term(answer["X"])
        except StopIteration:
            return None

    def is_terminal(self, symbol: str):
        return symbol in self.terminals
