"""
This module is used to compute node by queriyng a prolog engine running 
in the back. 
"""

import os
from typing import List, Union
import random
from lm_heuristic.tree import Node
from lm_heuristic.utils.prolog import PrologGrammarEngine


class PrologGrammarNode(Node):
    """
    Grammar node that use a prolog engine to compute :
    - the children of derivation
    - a random sentence that can be derivate from current node
    """

    def __init__(self, symbols: List[str], prolog_engine: PrologGrammarEngine):
        """
        :param symbols ordered sequences of symbol representing the current derivation string
        :param prolog_engine: interface toward Prolog
        """
        Node.__init__(self)
        self.prolog_engine = prolog_engine
        self._children: List["PrologGrammarNode"] = []
        self.symbols = symbols

    @classmethod
    def from_string(
        cls, prolog_engine: PrologGrammarEngine, str_nltk_grammar: str
    ) -> "PrologGrammarNode":
        """
        1/ Parse the grammar into prolog predicates.
        2/ Load the knowledge into the prolog engine.
        3/ Return the node corresponding to the start symbol
        """
        prolog_engine.load_grammar(str_nltk_grammar)

        return cls(["s"], prolog_engine)

    @classmethod
    def from_cfg_file(
        cls, prolog_engine: PrologGrammarEngine, path: str
    ) -> "PrologGrammarNode":
        """
        :param path: path toward a file containing the grammar
        """
        assert os.path.exists(path)
        with open(path) as file:
            str_grammar = file.read()
        return PrologGrammarNode.from_string(prolog_engine, str_grammar)

    def is_terminal(self) -> bool:
        for symbol in self.symbols:
            if not self.prolog_engine.is_terminal(symbol):
                return False
        return True

    def children(self) -> List["PrologGrammarNode"]:  # type: ignore
        # Note that we only return valid children !
        if self._children == []:
            self._children = [
                PrologGrammarNode(child_symbols, self.prolog_engine)
                for child_symbols in self.prolog_engine.valid_children(list(self.symbols))
            ]
        return self._children

    def random_children(self) -> "PrologGrammarNode":
        """
        return a random children
        """
        return random.choice(self.children())

    def random_walk(self) -> Union["PrologGrammarNode", None]:
        leaf_symbols = self.prolog_engine.leaf(self.symbols)
        if leaf_symbols:
            return PrologGrammarNode(leaf_symbols, self.prolog_engine)
        else:
            return None

    def __str__(self):
        return " ".join(map(str, self.symbols)) + "."

    def __hash__(self):
        return hash(self.__str__())
