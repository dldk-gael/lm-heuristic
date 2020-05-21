from flask import Flask
from flask_cors import CORS

from lm_heuristic.tree_search import RandomSearch, MonteCarloTreeSearch
from lm_heuristic.tree import CFGrammarNode
from lm_heuristic.sentence_score import GPT2Score
from lm_heuristic.heuristic import Heuristic
from lm_heuristic.generation import GPT2Paraphrases

# FLASK SERVER
app = Flask(__name__)
CORS(app)

# PARAPHRASING MODEL (ONLY LOAD ONCE IN MEMORY BECAUSE IT IS BIG)
PATH_TO_PARAPHRASE_CONTEXT = "data/text/paraphrase.txt"
GPT2_MODEL_TO_USE = "gpt2"
BATCH_SIZE = 1

with open(PATH_TO_PARAPHRASE_CONTEXT, "r") as file:
    paraphrase_context = file.read()

paraphrase_generator = GPT2Paraphrases(
    GPT2_MODEL_TO_USE, paraphasing_context=paraphrase_context, question_paraphrasing=False, batch_size=BATCH_SIZE,
)

# SEARCHER FOR GRAMMAR SAMPLING
no_heuristic = Heuristic(lambda terminal_nodes: [0] * len(terminal_nodes))
random_searcher = RandomSearch(no_heuristic)
montecarlo_searcher = MonteCarloTreeSearch(no_heuristic)
