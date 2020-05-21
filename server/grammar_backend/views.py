from flask import request, jsonify, Blueprint, url_for
from grammar_backend import app, random_searcher, montecarlo_searcher, celery
from .tasks import compute_paraphrase
from lm_heuristic.generation import generate_from_cfg
from lm_heuristic.tree import CFGrammarNode

views = Blueprint("views", __name__)


@app.route("/ping", methods=["GET"])
def ping():
    return "pong"

@app.route("/abort/paraphrase/<task_id>", methods=["GET"])
def abort_paraphrase(task_id):
    celery.control.revoke(task_id, terminate=True)
    return "paraphrase abort"

@app.route("/status/paraphrase/<task_id>", methods=["GET"])
def paraphrase_status(task_id):
    paraphrases = []
    task = compute_paraphrase.AsyncResult(task_id)
    if task.state == "SUCCESS":
        paraphrases = task.get()

    return jsonify({"status": task.state, "paraphrases":paraphrases})

@app.route("/paraphrase", methods=["POST"])
def paraphrase():
    data = request.get_json()
    task = compute_paraphrase.delay(data)
    return {'status_location': url_for('paraphrase_status', task_id=task.id),
            'abort_location': url_for('abort_paraphrase', task_id=task.id)}

@app.route("/grammar_sampling", methods=["POST"])
def grammar_sampling():
    print("here")
    data = request.get_json()
    grammar_root = CFGrammarNode.from_string(data["grammar"])
    if data["strategy"] == "MCTS":
        generations = generate_from_cfg(grammar_root, montecarlo_searcher, data["number_of_samples"], data["keep_top"])
    elif data["strategy"] == "Random Sampling":
        generations = generate_from_cfg(
            grammar_root, random_searcher, data["number_of_samples"], data["number_of_samples"]
        )
    return jsonify({"generations": generations})
