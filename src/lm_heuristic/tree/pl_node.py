import os
from typing import Tuple, Union, List
import random
from nltk.grammar import CFG, FeatureGrammar
from lm_heuristic.tree import Node
from lm_heuristic.prolog import PrologGrammarEngine


class PrologGrammarNode(Node):
    def __init__(self, symbols, prolog_engine: PrologGrammarEngine):
        """
        Only use if prologe_engine is loaded with a grammar
        """
        Node.__init__(self)
        self.prolog_engine = prolog_engine
        self.symbols = tuple(symbols) if not isinstance(symbols, tuple) else symbols

    @classmethod
    def from_string(
        cls, prolog_engine: PrologGrammarEngine, str_grammar: str, feature_grammar: bool = False
    ) -> "PrologGrammarNode":
        if feature_grammar:
            str_nltk_grammar = str(FeatureGrammar.fromstring(str_grammar))
        else:
            str_nltk_grammar = str(CFG.fromstring(str_grammar))

        prolog_engine.load_grammar(str_nltk_grammar)

        return cls("s", prolog_engine)

    @classmethod
    def from_cfg_file(
        cls, prolog_engine: PrologGrammarEngine, path: str, feature_grammar: bool = False
    ) -> "PrologGrammarNode":
        """
        :param path: path to file containing a context-free grammar
        :return: new Derivation tree node
        """
        assert os.path.exists(path)
        with open(path) as file:
            str_grammar = file.read()
        return PrologGrammarNode.from_string(prolog_engine, str_grammar, feature_grammar)

    def is_terminal(self) -> bool:
        for symbol in self.symbols:
            if not self.prolog_engine.is_terminal(symbol):
                return False
        return True

    def childrens(self) -> List["PrologGrammarNode"]:  # type: ignore
        # Note that we only return valid children !
        return [
            PrologGrammarNode(tuple(child_symbols), self.prolog_engine)
            for child_symbols in self.prolog_engine.valid_children(list(self.symbols))
        ] 

    def random_children(self) -> "PrologGrammarNode":
        """
        return a random children
        """
        #raise ValueError("random_children should not be use with PrologNode")
        _childrens = self.childrens()
        return random.choice(_childrens) if _childrens is not None else None

    def random_walk(self):
        leaf_symbols = self.prolog_engine.leaf(self.symbols)
        if leaf_symbols:
            return PrologGrammarNode(tuple(leaf_symbols), self.prolog_engine)
        else:
            return None # when there is no leaf from current node (will be useful for FeatureGrammar)

    def __str__(self):
        return " ".join(map(str, self.symbols)) + "."

    def __hash__(self):
        return hash(self.__str__())
