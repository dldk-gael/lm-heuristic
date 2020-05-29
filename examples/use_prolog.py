from nltk import CFG
from lm_heuristic.prolog import PrologGrammarEngine

if __name__ == "__main__":
    pl_grammar = PrologGrammarEngine()

    with open("data/cfg/bas.cfg") as file:
        str_grammar = file.read()
        ntlk_str_grammar = str(CFG.fromstring(str_grammar))

    pl_grammar.load_grammar(ntlk_str_grammar)
    print(pl_grammar.terminals)
    print(pl_grammar.children(['snp', 'vp']))