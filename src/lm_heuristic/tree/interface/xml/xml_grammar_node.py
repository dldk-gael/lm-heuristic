from copy import deepcopy
from lm_heuristic.tree.node import Node
from .parse_xml import xml_to_string
from .parse_grammar_str import parse_grammar
from .feature_structure import PStruct, is_symbol_terminal, copy_features, revert


class XMLGrammarNode(Node):
    def __init__(self, symbols: tuple, grammar, shrink=True):
        self.symbols = symbols
        self.grammar = grammar
        self._children = None
        self.shrink = shrink

    @classmethod
    def from_file(cls, file, **kwargs):
        grammar_as_str = xml_to_string(file)
        grammar = parse_grammar(grammar_as_str)
        return cls(({"str": "ROOT", "features": PStruct({})},), grammar, **kwargs)

    def __hash__(self):
        return hash(str(self)) # TODO optimize that

    def __str__(self):
        if self.is_terminal():
            as_str = ""
            for symbol in self.symbols:
                if not len(symbol["str"]) == 2: # to remove "" or ''
                    str_symbol = symbol["str"][1:-1] # to remove " " from "word"
                    if as_str != "":
                        as_str += " " + str_symbol
                    else:
                        as_str += str_symbol

            return as_str
        else: # for debug only
            return " ".join(map(str, self.symbols))

    def compute_children(self):
        child_nodes = []

        # Go from left to right to the first non terminal symbol
        idx_left_nt_symb = 0
        for symbol in self.symbols:
            if not is_symbol_terminal(symbol):
                break
            idx_left_nt_symb += 1
        symbol = self.symbols[idx_left_nt_symb]

        productions = self.grammar.get(symbol["str"], [])

        if not productions:
            # Error in grammar -> miss a non terminal symbol
            return [XMLGrammarNode(({"str": "DEAD_END", "features": PStruct({})},), self.grammar)]

        for production in productions:
            feature_copies = copy_features(production)
            head_bindings = []
            if not symbol["features"].unify(feature_copies[0]["features"], head_bindings):
                revert(head_bindings)
                continue

            new_node = XMLGrammarNode(
                symbols=deepcopy(self.symbols[:idx_left_nt_symb] + tuple(feature_copies[1:]) + self.symbols[idx_left_nt_symb + 1 :]),
                grammar=self.grammar,
            )
            child_nodes.append(new_node)
            revert(head_bindings)

        return child_nodes

    def children(self):
        if not self._children:
            self._children = self.compute_children()
            if (len(self._children) == 1) and (not self._children[0].is_terminal()) and (self.shrink):
                self._children = self._children[0].children()
            
        return self._children

    def is_terminal(self):
        for symbol in self.symbols:
            if not is_symbol_terminal(symbol):
                return False
        return True
