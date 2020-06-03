import os
from io import StringIO

from transformers import GPT2LMHeadModel
import tensorflow_hub as hub
from nltk.parse import CoreNLPParser
from nltk.tree import Tree

from lm_heuristic.generation import GPT2Paraphrases, generate_from_cfg
from lm_heuristic.prolog import PrologGrammarEngine
from lm_heuristic.tree import PrologGrammarNode
from lm_heuristic.sentence_score import GPT2Score
from lm_heuristic.tree_search import MonteCarloTreeSearch, RandomSearch, AllocationStrategy
from lm_heuristic.heuristic import Heuristic

from grammar_backend import celery

# INITIALIZATION

PATH_TO_PARAPHRASE_CONTEXT = "../data/text/paraphrase.txt"
PATH_TO_UNIVERSAL_SENTENCE_ENCODER = "../model_weights/universal_sentence_encoder/"
GPT2_NAME = "gpt2"
BATCH_SIZE = 1
GPT2_MODEL = None
USE_MODEL = None
PARAPHRASE_GENERATOR = None
MONTECARLO_SEARCHER = None
RANDOM_SEARCHER = None
PROLOG_ENGINE = None
PARSER = None


def load_gpt2():
    global GPT2_MODEL
    GPT2_MODEL = GPT2LMHeadModel.from_pretrained(GPT2_NAME)


def load_use():
    global USE_MODEL
    assert os.path.exists(PATH_TO_UNIVERSAL_SENTENCE_ENCODER), "USE weights was not found."
    USE_MODEL = hub.load(PATH_TO_UNIVERSAL_SENTENCE_ENCODER)


def initialize_paraphraser():
    global PARAPHRASE_GENERATOR
    if not GPT2_MODEL:
        load_gpt2()
    if not USE_MODEL:
        load_use()

    with open(PATH_TO_PARAPHRASE_CONTEXT, "r") as file:
        paraphrase_context = file.read()

    PARAPHRASE_GENERATOR = GPT2Paraphrases(
        gpt2_model_name=GPT2_NAME,
        gpt2_model=GPT2_MODEL,
        paraphasing_context=paraphrase_context,
        question_paraphrasing=False,
        sentence_encoder=USE_MODEL,
        batch_size=BATCH_SIZE,
    )


def initialize_MCTS():
    global MONTECARLO_SEARCHER
    if not GPT2_MODEL:
        load_gpt2()

    gpt_2_scorer = GPT2Score(model_name=GPT2_NAME, model=GPT2_MODEL, batch_size=BATCH_SIZE, length_normalization=True)
    evaluation_fn = lambda terminal_nodes: gpt_2_scorer(list(map(str, terminal_nodes)))
    heuristic = Heuristic(evaluation_fn, use_memory=True)

    # Initialize the search parameters
    MONTECARLO_SEARCHER = MonteCarloTreeSearch(
        heuristic=heuristic,
        buffer_size=BATCH_SIZE,
        c=1,
        d=1000,
        t=0,
        allocation_strategy=AllocationStrategy.UNIFORM,
        verbose=True,
    )


def initialize_random_search():
    global RANDOM_SEARCHER
    RANDOM_SEARCHER = RandomSearch(Heuristic(lambda terminal_nodes: [0] * len(terminal_nodes)))


def initialize_prolog_engine():
    global PROLOG_ENGINE
    PROLOG_ENGINE = PrologGrammarEngine("../prolog/methods.pl")

def initialize_parser():
    global PARSER
    PARSER = CoreNLPParser(url='http://localhost:9000')


# TASK DEFINITION
@celery.task(bind=True, name="compute_paraphrase")
def compute_paraphrase(self, data):
    print("%s" % data["sentences_to_paraphrase"])
    if not PARAPHRASE_GENERATOR:
        self.update_state(state="PROGRESS", meta={"detail": "Loading langage model ..."})
        initialize_paraphraser()

    self.update_state(state="PROGRESS", meta={"detail": "Generating paraphrases ..."})
    paraphrases = PARAPHRASE_GENERATOR.paraphrase_multiple_sentences(
        sentences=data["sentences_to_paraphrase"],
        forbidden_words=data["forbidden_words"],
        nb_samples_per_sentence=data["number_of_samples"],
        top_n_to_keep_per_sentence=data["keep_top"],
    )
    return paraphrases


@celery.task(bind=True, name="grammar_random_search")
def grammar_random_search(self, data):
    if not PROLOG_ENGINE:
        self.update_state(state="PROGRESS", meta={"detail": "Loading Prolog ..."})
        initialize_prolog_engine()
    PROLOG_ENGINE.delete_grammar() # delete existing grammar that are in memory (could be optimize latter)

    if not RANDOM_SEARCHER:
        self.update_state(state="PROGRESS", meta={"detail": "Initializing random searcher ..."})
        initialize_random_search()

    self.update_state(state="PROGRESS", meta={"detail": "Random sampling ..."})
    grammar_root = PrologGrammarNode.from_string(PROLOG_ENGINE, data["grammar"])
    generations = generate_from_cfg(
        grammar_root, RANDOM_SEARCHER, data["number_of_samples"], data["number_of_samples"]
    )
    return generations


@celery.task(bind=True, name="grammar_mcts")
def grammar_mcts(self, data):
    if not PROLOG_ENGINE:
        self.update_state(state="PROGRESS", meta={"detail": "Loading Prolog ..."})
        initialize_prolog_engine()
    PROLOG_ENGINE.delete_grammar() # delete existing grammar that are in memory (could be optimize latter)

    if not MONTECARLO_SEARCHER:
        self.update_state(state="PROGRESS", meta={"detail": "Loading langage model ..."})
        initialize_MCTS()

    self.update_state(state="PROGRESS", meta={"detail": "Perfoming the tree walks ..."})
    grammar_root = PrologGrammarNode.from_string(PROLOG_ENGINE, data["grammar"])
    generations = generate_from_cfg(grammar_root, MONTECARLO_SEARCHER, data["number_of_tree_walks"], data["keep_top"])
    return generations

@celery.task(name="parse_tree")
def parse_tree(sentence):
    if not PARSER:
        initialize_parser()

    tree = next(PARSER.raw_parse(sentence))
    output = StringIO()
    Tree.fromstring(str(tree)).pretty_print(stream=output)
    tree_str = output.getvalue()

    return tree_str