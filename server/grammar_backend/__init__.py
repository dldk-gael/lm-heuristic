from flask import Flask
from flask_cors import CORS
from celery import Celery

# FLASK SERVER
app = Flask(__name__)
CORS(app)

# CELERY WORKER
redis_URI = "redis://localhost:6379"
celery = Celery(app.import_name, backend=redis_URI, broker=redis_URI, include="grammar_backend.tasks")
celery.conf.update(app.config)
celery.conf.update(
    {
        "task_routes": {
            "compute_paraphrase": {"queue": "language_model"},
            "grammar_mcts": {"queue": "language_model"},
            "grammar_random_search": {"queue": "default"},
        },
    }
)

