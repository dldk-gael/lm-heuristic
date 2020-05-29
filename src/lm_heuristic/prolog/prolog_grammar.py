import os
from typing import List
from pyswip import Prolog
from lm_heuristic.prolog.utils import convert_grammar_to_prolog


class PrologGrammarEngine():
    all_ready_initialize = False

    def __init__(self):
        """
        Must be initialize only once.
        """
        assert not PrologGrammarEngine.all_ready_initialize
        self.prolog = Prolog()
        self.prolog.consult(os.path.dirname(os.path.realpath(__file__)) + "/methods.pl")

        self.current_predicates = []
        self.terminals = {}
        PrologGrammarEngine.all_ready_initialize = True

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
        self.retrieve_terminal()
    
    @staticmethod
    def format_answer(answer):
        return [str(var) for var in answer]

    def children(self, symbols: List[str]):
        answers = self.prolog.query("child([%s], X)" % ", ".join(symbols))
        return [self.format_answer(answer["X"]) for answer in answers]

    def all_leaf(self, symbols: List[str]):
        answers = self.prolog.query("leaf([%s], X)" % ", ".join(symbols))
        return [self.format_answer(answer["X"]) for answer in answers]
 
    def leaf(self, symbols: List[str]):
        answers = self.prolog.query("leaf([%s], X)" % ", ".join(symbols))
        # TODO : add random here. How ?!
        return self.format_answer(next(answers)["X"])

    def is_terminal(self, symbol: str):
        return symbol in self.terminals