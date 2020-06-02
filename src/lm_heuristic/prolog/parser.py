from typing import List, Dict, Union, Set
from nltk.grammar import FeatStructNonterminal, Production, FeatureGrammar, CFG, Nonterminal
from nltk.featstruct import Feature
from nltk.sem import Variable


class ParseToProlog:
    def __init__(self, feature_grammar: bool = False):
        self.feature_grammar = feature_grammar
        self.terminal_symbols: Set[str] = set()
        if self.feature_grammar:
            # features_dict: symbol -> list of all feature use with these symbol in the grammar
            self.features_dict: Dict[str, List[str]] = dict()

    def __call__(self, grammar_str) -> List[str]:
        self.features_dict = dict()
        self.terminal_symbols = set()

        if self.feature_grammar:
            return self._parse_feature_str_grammar(grammar_str)
        else:
            return self._parse_CFG_str_grammar(grammar_str)

    def _compute_features_dict(self, productions: List[Production]):
        for production in productions:
            for term in (production.lhs(), ) + production.rhs():
                if isinstance(term, Nonterminal):
                    name = str(term[Feature("type")]).lower()
                    self.features_dict.setdefault(name, [])
                    current_features = set(self.features_dict[name])
                    new_features = set(str(feature) for feature in term if str(feature) != "*type*")
                    self.features_dict[name] = list(current_features.union(new_features))
                else:
                    self.terminal_symbols.add(term.lower())

    @staticmethod
    def _transform_value(value):
        if isinstance(value, Variable):
            # Then, the variable is on the form ?x_1
            # We transform that to X_1
            return str(value)[1:].upper()
        else:
            return str(value).lower()

    def _parse_term(self, term: Union[FeatStructNonterminal, str]) -> str:
        """
        Use to parse feature grammar that respect NLTK FeatGrammar format
        """
        if not isinstance(term, Nonterminal):
            return term.lower()

        name = str(term[Feature("type")]).lower()
        param_list = []
        for feature in self.features_dict[name]:
            if feature in term:
                param_list.append(self._transform_value(term[feature]))
            else:
                param_list.append("_")
        return "%s(%s)"%(name, ", ".join(param_list)) if len(param_list) != 0 else name

    def _parse_feature_str_grammar(self, grammar_str: str) -> List[str]:
        nltk_featgrammar = FeatureGrammar.fromstring(grammar_str)
        self._compute_features_dict(nltk_featgrammar.productions())

        pl_predicates = []

        for production in nltk_featgrammar.productions():
            lhs = self._parse_term(production.lhs())
            rhs = [self._parse_term(term) for term in production.rhs()]
            pl_predicates.append("rule(%s, [%s])" % (lhs, ", ".join(rhs)))

        for terminal in self.terminal_symbols:
            pl_predicates.append("terminal(%s)" % terminal.lower())
        
        return pl_predicates
    
    def _parse_CFG_str_grammar(self, grammar_str: str) -> List[str]:
        """
        Use to parse grammar that as the form
            # Rules
            s -> np vp
            np -> n | det n
            ...
            # Lexicon
            v -> 'want' | 'wants'

        Into a list of prolog predicates
            rule(s, [np, vp])
            rule(np, [n])
            rule(np, [det, n])
            ...
            rule(v, ['want'])
            terminal('want)
        """
        nltk_grammar = CFG.fromstring(grammar_str)
        pl_predicates = []
        for production in nltk_grammar.productions():
            lhs = str(production.lhs()).lower()
            rhs = []
            for symb in production.rhs():
                rhs.append(str(symb).lower())
                if not isinstance(symb, Nonterminal):
                    self.terminal_symbols.add(str(symb).lower())
            rhs = [str(symb).lower() for symb in production.rhs()]
            pl_predicates.append("rule(%s, [%s])" % (lhs, ", ".join(rhs)))
    


        for terminal in self.terminal_symbols:
            pl_predicates.append("terminal(%s)" % terminal)

        return pl_predicates