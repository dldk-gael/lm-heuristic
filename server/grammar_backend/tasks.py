from io import StringIO

from nltk.tree import Tree

from lm_heuristic.generation import generate_from_grammar
from lm_heuristic.tree.interface.nltk_grammar import FeatureGrammarNode

from grammar_backend import celery
from grammar_backend.models import Models

models = Models()


@celery.task(bind=True, name="compute_paraphrase")
def compute_paraphrase(self, data):
    if not models.is_paraphrase_generator_ready():
        self.update_state(state="PROGRESS", meta={"detail": "Loading langage model ..."})
        models.load_paraphrase_generator()

    self.update_state(state="PROGRESS", meta={"detail": "Generating paraphrases ..."})
    return models.paraphrase_generator.paraphrase_multiple_sentences(
        sentences=data["sentences_to_paraphrase"],
        forbidden_words=data["forbidden_words"],
        nb_samples_per_sentence=data["number_of_samples"],
        top_n_to_keep_per_sentence=data["keep_top"],
    )


@celery.task(bind=True, name="grammar_random_search")
def grammar_random_search(self, data):
    self.update_state(state="PROGRESS", meta={"detail": "Random sampling ..."})
    grammar_root = FeatureGrammarNode.from_string(data["grammar"])
    random_valid_leaves = []
    for _ in range(data["number_of_samples"]):
        new_valid_leaf = grammar_root.find_random_valid_leaf()
        if not new_valid_leaf:
            return ["NO SOLUTION FOUND"]
        else:
            random_valid_leaves.append(str(new_valid_leaf))

    return random_valid_leaves


@celery.task(bind=True, name="grammar_mcts")
def grammar_mcts(self, data):
    if not models.is_montecarlo_searcher_ready():
        self.update_state(state="PROGRESS", meta={"detail": "Loading langage model ..."})
        models.load_montecarlo_searcher()

    self.update_state(state="PROGRESS", meta={"detail": "Perfoming the tree walks ..."})
    return generate_from_grammar(
        grammar_root=FeatureGrammarNode.from_string(data["grammar"]),
        searcher=models.montecarlo_searcher,
        nb_tree_walks=data["number_of_tree_walks"],
        keep_top_n=data["keep_top"],
    )


@celery.task(name="parse_tree")
def parse_tree(sentence):
    if not models.is_parser_ready():
        models.load_parser()

    tree = next(models.parser.raw_parse(sentence))
    output = StringIO()
    Tree.fromstring(str(tree)).pretty_print(stream=output)
    tree_str = output.getvalue()

    return tree_str
