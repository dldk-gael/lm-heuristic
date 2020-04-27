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
    def __init__(self, items: tuple,  cfg: CFG):
        """
        Initalize a Derivation Node
        :param items ordered sequences of symbol representing the current derivation string
        :param cfg : reference to context free grammar containing the production rules
        """
        Node.__init__(self)
        self.cfg = cfg
        self.items = (items, ) if type(items) != tuple else items

    def is_terminal(self) -> bool:
        """
        return True if the current derivation string is only composed of terminal symbols
        """
        for item in self.items:
            if type(item) == grammar.Nonterminal:
                return False
        return True

    def childrens(self) -> List[Node]:
        """
        return all the Derativation that derivated from the current derivation using only one production rule from P
        """
        childrens = []
        for idx, item in enumerate(self.items):
            if type(item) == grammar.Nonterminal:
                productions = self.cfg.productions(lhs=item)
                for production in productions:
                    childrens.append(Derivation(self.items[:idx] + production.rhs() + self.items[idx+1:], self.cfg))
        return childrens

    def random_children(self):
        #  TODO optimize this function
        return random.choice(self.childrens())

    def __str__(self):
        return " ".join(map(str, self.items)) + "."
