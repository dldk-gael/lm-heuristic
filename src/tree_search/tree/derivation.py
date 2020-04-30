from tree_search.tree import Node
from nltk import CFG, grammar
from typing import List
import random


class Derivation(Node):
    """
    Given a context-free grammar (T, N, P, S) :
    - T: a set of terminal symbol
    - N: a set of non-terminal symbol
    - P: a set of production rules
    - S: the start symbol (belong to N)

    A derivation is a string over T union N that can be derivate from S using production rules from P
    Derivation is used to embed a particular derivation string as a tree node.
    The childrens of one node consist of all the string that derivated from the current derivation using only
    one production rule from P
    """
    def __init__(self, symbols: tuple,  cfg: CFG, shrink=False):
        """
        Initalize a Derivation Node
        :param symbols ordered sequences of symbol representing the current derivation string
        :param cfg : reference to context free grammar containing the production rules
        :param shrink: bool, if true when asking for the childrens and there is only one,
                                will directly return grandchildrens
        """
        Node.__init__(self)
        self.cfg = cfg
        self.symbols = (symbols, ) if type(symbols) != tuple else symbols
        self.shrink = shrink

    def is_terminal(self):
        """
        return True if the current derivation string is only composed of terminal symbols
        """
        for symbol in self.symbols:
            if type(symbol) == grammar.Nonterminal:
                return False
        return True

    def childrens(self):
        """
        return all the Derivations nodes corresponding to sentences that can be derivated f
        from the current derivation using only one production rule from P
        if shrink option has been selected, will directly grand childrens if there is only one single children
        """
        childrens = []
        for idx, symbol in enumerate(self.symbols):
            if type(symbol) == grammar.Nonterminal:
                productions = self.cfg.productions(lhs=symbol)
                for production in productions:
                    childrens.append(Derivation(self.symbols[:idx] + production.rhs() + self.symbols[idx+1:], self.cfg))

        if self.shrink and len(childrens) == 1:
            return childrens[0].childrens()

        return childrens

    def random_children(self):
        return random.choice(self.childrens())

    def __str__(self):
        return " ".join(map(str, self.symbols)) + "."
