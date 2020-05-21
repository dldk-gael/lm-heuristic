from flask import Flask
from flask_cors import CORS
from celery import Celery

from lm_heuristic.tree_search import RandomSearch, MonteCarloTreeSearch
from lm_heuristic.tree import CFGrammarNode
from lm_heuristic.sentence_score import GPT2Score
from lm_heuristic.heuristic import Heuristic
from lm_heuristic.generation import GPT2Paraphrases

# FLASK SERVER
app = Flask(__name__)
CORS(app)

# CELERY WORKER
redis_URI = "redis://localhost:6379"
celery = Celery(app.import_name, backend=redis_URI, broker=redis_URI, include="grammar_backend.tasks")
celery.conf.update(app.config)

# SEARCHER FOR GRAMMAR SAMPLING
no_heuristic = Heuristic(lambda terminal_nodes: [0] * len(terminal_nodes))
random_searcher = RandomSearch(no_heuristic)
montecarlo_searcher = MonteCarloTreeSearch(no_heuristic)

