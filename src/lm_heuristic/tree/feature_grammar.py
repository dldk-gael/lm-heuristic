import os
from typing import *
from functools import wraps
import random

from nltk.grammar import FeatureGrammar, Nonterminal, FeatStructNonterminal
from nltk.featstruct import find_variables, unify, rename_variables, substitute_bindings
from nltk.sem import Variable

from .node import Node


# The rename_variables and substitute_bindings from NLTK can not be applyed to raw string
# We modify them in order that to handle raw string (they simply directly it) 
def skip_terminal_symbole(function):
    @wraps(function)
    def function_skipping_terminal_symbol(symbol, *args, **kwargs):
        if isinstance(symbol, str):
            return symbol
        else:
            return function(symbol, *args, **kwargs)

    return function_skipping_terminal_symbol


substitute_bindings = skip_terminal_symbole(substitute_bindings)
rename_variables = skip_terminal_symbole(rename_variables)


class FeatureGrammarNode(Node):
    def __init__(
        self,
        symbols: Union[Tuple[FeatStructNonterminal], FeatStructNonterminal],
        feature_grammar: FeatureGrammar,
        only_keep_valid_node: bool = False,
    ):
        """
        :param symbols ordered sequences of symbol representing the current derivation string
        :param feature_grammar : reference to ntlk feature grammar containing the production rules
        :param only_keep_valid_node: bool, if true when asking for the children, will filter the one from which no valid leaf can be reached
        """
        Node.__init__(self)
        self.feature_grammar = feature_grammar
        self.symbols = (symbols,) if not isinstance(symbols, tuple) else symbols
        self._children: List["FeatureGrammarNode"]
        self._children_have_been_computed = False
        self.only_keep_valid_node = only_keep_valid_node

    @classmethod
    def from_string(cls, str_grammar: str, **kwargs) -> "FeatureGrammarNode":
        feature_grammar = FeatureGrammar.fromstring(str_grammar)
        return cls(feature_grammar.start(), feature_grammar, **kwargs)

    @classmethod
    def from_fcfg_file(cls, path: str, **kwargs) -> "FeatureGrammarNode":
        """
        :param path: path to file containing a context-free grammar
        :return: new Derivation tree node
        """
        assert os.path.exists(path)
        with open(path) as file:
            str_grammar = file.read()
        return cls.from_string(str_grammar, **kwargs)

    def children(self) -> List["FeatureGrammarNode"]:  # type: ignore
        if not self._children_have_been_computed:
            non_filter_children = self.compute_children()
            if self.only_keep_valid_node:
                self._children = [child for child in non_filter_children if child.find_random_valid_leaf()]
            else:
                self._children = non_filter_children
            self._children_have_been_computed = True

        return self._children

    def compute_children(self) -> List["FeatureGrammarNode"]:
        child_list: List["FeatureGrammarNode"] = []

        # First we retrieve all variables using in current derivation
        used_vars: Set[Variable] = set()
        for symbol in self.symbols:
            if not isinstance(symbol, str):
                used_vars |= find_variables(symbol)

        for idx, symbol in enumerate(self.symbols):
            if isinstance(symbol, str):
                continue

            # For each non terminal symbol in current derivation , we select a production rule
            # that has a left hand side matching this symbol
            for production in self.feature_grammar.productions(lhs=symbol):

                # We rename all the variable in the production rules to avoid name conflicts
                # TODO put this after a check to avoid to do it if not neccessary
                new_vars = dict()
                lhs = rename_variables(production.lhs(), used_vars=used_vars, new_vars=new_vars)
                rhs = [
                    rename_variables(rhs_symb, used_vars=used_vars, new_vars=new_vars) for rhs_symb in production.rhs()
                ]

                # Compute the new binding
                new_bindings = dict()
                lhs = unify(lhs, symbol, bindings=new_bindings)
                if lhs is None:  # Unification failed
                    continue

                # Propagate the bindings to the siblings
                new_siblings = [substitute_bindings(sibling, bindings=new_bindings) for sibling in self.symbols]

                # Propagate the bindings to the rhs symbols
                new_rhs = [substitute_bindings(rhs_symb, bindings=new_bindings) for rhs_symb in rhs]

                # Create the new child
                new_child = FeatureGrammarNode(
                    tuple(new_siblings[:idx] + new_rhs + new_siblings[idx + 1 :]), self.feature_grammar,
                )
                child_list.append(new_child)

        return child_list if len(child_list) != 0 else [FeatureGrammarNode("DEAD_END", self.feature_grammar)]


    def find_random_valid_leaf(self):
        leaves = []
        if str(self) == "DEAD_END.":
            return None

        for symbol in self.symbols:
            if not isinstance(symbol, Nonterminal):
                leaves.append(symbol)
                continue

            children_symbol = FeatureGrammarNode(symbol, self.feature_grammar).children()
            random.shuffle(children_symbol)
            leaf = None
            for child in children_symbol:
                leaf = child.find_random_valid_leaf()
                if leaf: 
                    break
            if not leaf:
                return None 
            else:
                leaves.append(leaf)
        return " ".join(leaves)

    def find_random_valid_leaf_debug(self) -> Union["FeatureGrammarNode", None]:
        # This version is much more slower but make it easy to debug grammar
        if self.is_terminal():
            return self

        if str(self) == "DEAD_END.":
            return None

        node_children = self.children()
        shuffle_index = list(range(len(node_children)))
        random.shuffle(shuffle_index)

        for i in shuffle_index:
            leaf = node_children[i].find_random_valid_leaf_debug()
            if leaf and str(leaf) != "DEAD_END.":
                return leaf

        return None

    def __hash__(self):
        return hash(self.symbols)

    def __str__(self):
        if self.is_terminal():
            return " ".join(self.symbols) + "."
        else:  # for debug mode mainly
            return "\n".join(map(str, self.symbols))

    def is_terminal(self):
        for symbol in self.symbols:
            if isinstance(symbol, Nonterminal):
                return False
        return True
