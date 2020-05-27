from typing import List, Tuple, Dict, Any
import os
from nltk.grammar import FeatureGrammar, Nonterminal
from nltk.sem import Variable
from .node import Node


class FeatureGrammarNode(Node):
    variable_counter = 0

    def __init__(self, symbols: tuple, feature_grammar: FeatureGrammar):
        Node.__init__(self)
        self.feature_grammar = feature_grammar
        self.symbols = (symbols,) if not isinstance(symbols, tuple) else symbols

    @classmethod
    def from_cfg_file(cls, path: str, **kwargs) -> "FeatureGrammarNode":
        """
        :param path: path to file containing a context-free grammar
        :return: new Derivation tree node
        """
        assert os.path.exists(path)
        with open(path) as file:
            str_grammar = file.read()
        feature_grammar = FeatureGrammar.fromstring(str_grammar)
        return cls(feature_grammar.start(), feature_grammar)

    def _new_variable(self):
        FeatureGrammarNode.variable_counter += 1
        return Variable("?x_" + str(FeatureGrammarNode.variable_counter))

    def _new_binding_to_propagate(self, key, parent, child):
        """
        Return true if the new chosen child symbol implies a binding for key wrt to the parent
        """
        return (key in child) and (not isinstance(child[key], Variable)) and (isinstance(parent[key], Variable))

    def _compute_news_bindings(self, parent, child):
        var_bindings = {
            parent[key]: child[key] for key in parent if self._new_binding_to_propagate(key, parent, child)
        }
        return var_bindings

    def _update_bindings(self, symbols, var_bindings):
        new_symbols = []
        for symbol in symbols:
            if isinstance(symbol, Nonterminal):
                new_symbol = symbol.copy(deep=True).substitute_bindings(var_bindings)
            else:
                new_symbol = symbol
            new_symbols.append(new_symbol)
        return tuple(new_symbols)

    def _update_lhs(self, lhs_symbol, parent_symbol):
        symbol = lhs_symbol.copy(deep=True)
        for key in symbol:
            if key in parent_symbol and not self._new_binding_to_propagate(key, parent_symbol, symbol):
                symbol[key] = parent_symbol[key]
            elif isinstance(symbol[key], Variable):
                symbol[key] = self._new_variable()
        return symbol

    def _update_rhs(self, lhs_symbol, rhs_symbols):
        new_variables: Dict[Variable, Variable] = {}
        rhs = []
        for rhs_symbol in rhs_symbols:
            if isinstance(rhs_symbol, Nonterminal):
                new_rhs_symbol = rhs_symbol.copy(deep=True)
                for key in new_rhs_symbol:
                    if key in lhs_symbol and str(key) != "*type*":
                        new_rhs_symbol[key] = lhs_symbol[key]
                    elif isinstance(new_rhs_symbol[key], Variable):
                        variable = new_rhs_symbol[key]
                        if not variable in new_variables:
                            new_variables[variable] = self._new_variable()
                        new_rhs_symbol[key] = new_variables[variable]
                rhs.append(new_rhs_symbol)
            else:
                rhs.append(rhs_symbol)
        return tuple(rhs)

    def childrens(self):
        child_list = []
        for idx, symbol in enumerate(self.symbols):
            if isinstance(symbol, Nonterminal):
                for prod in self.feature_grammar.productions(lhs=symbol):
                    new_lhs = prod.lhs().unify(symbol)
                    if new_lhs is not None:
                        new_var_bindings = self._compute_news_bindings(symbol, new_lhs)
                        siblings = self._update_bindings(self.symbols, new_var_bindings)
                        new_lhs = self._update_lhs(new_lhs, symbol)
                        new_rhs = self._update_rhs(new_lhs, prod.rhs())

                        child_list.append(
                            FeatureGrammarNode(siblings[:idx] + new_rhs + siblings[idx + 1 :], self.feature_grammar)
                        )
        return child_list

    def __hash__(self):
        return hash(self.symbols)

    def __str__(self):
        if self.is_terminal():
            return " ".join(self.symbols) + "."
        else: # for debug mode mainly
            return "\n".join(map(str, self.symbols))

    def is_terminal(self):
        for symbol in self.symbols:
            if isinstance(symbol, Nonterminal):
                return False
        return True

    def random_children(self):
        pass
