from transformers import GPT2LMHeadModel
import tensorflow_hub as hub

from lm_heuristic.generation import GPT2Paraphrases, generate_from_cfg
from lm_heuristic.tree import CFGrammarNode
from lm_heuristic.sentence_score import GPT2Score
from lm_heuristic.tree_search import MonteCarloTreeSearch, RandomSearch, AllocationStrategy
from lm_heuristic.heuristic import Heuristic

from grammar_backend import celery

# INITIALIZATION 

PATH_TO_PARAPHRASE_CONTEXT = "../data/text/paraphrase.txt"
GPT2_NAME = "gpt2"
BATCH_SIZE = 1
GPT2_MODEL = None
USE_MODEL = None
PARAPHRASE_GENERATOR = None
MONTECARLO_SEARCHER = None
RANDOM_SEARCHER = None 


def load_gpt2():
    global GPT2_MODEL
    GPT2_MODEL = GPT2LMHeadModel.from_pretrained(GPT2_NAME)


def load_use():
    global USE_MODEL
    USE_MODEL = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")


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

# TASK DEFINITION

@celery.task(bind=True, name="compute_paraphrase")
def compute_paraphrase(self, data):
    self.update_state(state="PROGRESS", meta={"detail":"Loading langage model ..."})
    if not PARAPHRASE_GENERATOR:
        initialize_paraphraser()

    self.update_state(state="PROGRESS", meta={"detail":"Generating paraphrases ..."})
    paraphrases = PARAPHRASE_GENERATOR(
        sentence=data["sentence_to_paraphrase"],
        forbidden_words=data["forbidden_words"],
        nb_samples=data["number_of_samples"],
        top_n_to_keep=data["keep_top"],
    )
    return paraphrases


@celery.task(bind=True, name="grammar_random_search")
def grammar_random_search(self, data):
    self.update_state(state="PROGRESS", meta={"detail":"Initializing random searcher ..."})
    if not RANDOM_SEARCHER:
        initialize_random_search()


    self.update_state(state="PROGRESS", meta={"detail":"Random sampling ..."})
    grammar_root = CFGrammarNode.from_string(data["grammar"])
    generations = generate_from_cfg(
        grammar_root, RANDOM_SEARCHER, data["number_of_samples"], data["number_of_samples"]
    )
    return generations


@celery.task(bind=True, name="grammar_mcts")
def grammar_mcts(self, data):
    self.update_state(state="PROGRESS", meta={"detail":"Loading langage model ..."})
    if not MONTECARLO_SEARCHER:
        initialize_MCTS()

    self.update_state(state="PROGRESS",  meta={"detail":"Perfoming the tree walks ..."})
    grammar_root = CFGrammarNode.from_string(data["grammar"])
    generations = generate_from_cfg(
        grammar_root, MONTECARLO_SEARCHER, data["number_of_tree_walks"], data["keep_top"]
    )
    return generations
