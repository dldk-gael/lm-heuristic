from tree_search.tree import Node
from nltk import CFG, grammar
from typing import List


class Derivation(Node):
    def __init__(self, items: tuple,  cfg: CFG):
        Node.__init__(self)
        self.cfg = cfg
        self.items = (items, ) if type(items) != tuple else items

    def is_terminal(self) -> bool:
        for item in self.items:
            if type(item) == grammar.Nonterminal:
                return False
        return True

    def childrens(self) -> List[Node]:
        childrens = []
        for idx, item in enumerate(self.items):
            if type(item) == grammar.Nonterminal:
                productions = self.cfg.productions(lhs=item)
                for production in productions:
                    childrens.append(Derivation(self.items[:idx] + production.rhs() + self.items[idx+1:], self.cfg))
        return childrens

    def __hash__(self):
        return hash(self.items)

    def __str__(self):
        return " ".join(map(str, self.items))
