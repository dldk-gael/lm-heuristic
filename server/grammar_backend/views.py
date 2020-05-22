from flask import request, jsonify, Blueprint, url_for
from grammar_backend import app, celery
from .tasks import compute_paraphrase, grammar_random_search, grammar_mcts

views = Blueprint("views", __name__)


@app.route("/ping", methods=["GET"])
def ping():
    return "pong"

@app.route("/paraphrase", methods=["POST"])
def paraphrase():
    data = request.get_json()
    task = compute_paraphrase.delay(data)
    return {'status_location': url_for('paraphrase_status', task_id=task.id),
            'abort_location': url_for('abort_task', task_id=task.id)}

@app.route("/grammar/random_search", methods=["POST"])
def random_search():
    data = request.get_json()
    task = grammar_random_search.delay(data)
    return {'status_location': url_for('grammar_random_search_status', task_id=task.id),
            'abort_location': url_for('abort_task', task_id=task.id)}

@app.route("/grammar/mcts", methods=["POST"])
def mcts():
    data = request.get_json()
    task = grammar_mcts.delay(data)
    return {'status_location': url_for('grammar_mcts_status', task_id=task.id),
            'abort_location': url_for('abort_task', task_id=task.id)}


@app.route("/abort/<task_id>", methods=["GET"])
def abort_task(task_id):
    celery.control.revoke(task_id, terminate=True)
    return "TASK KILLED"

@app.route("/status/paraphrase/<task_id>", methods=["GET"])
def paraphrase_status(task_id):
    paraphrases = []
    task = compute_paraphrase.AsyncResult(task_id)
    if task.state == "SUCCESS":
        paraphrases = task.get()

    return jsonify({"status": task.state, "paraphrases":paraphrases})

@app.route("/status/grammar_random_search/<task_id>", methods=["GET"])
def grammar_random_search_status(task_id):
    generations = []
    task = grammar_random_search.AsyncResult(task_id)
    if task.state == "SUCCESS":
        generations = task.get()

    return jsonify({"status": task.state, "generations":generations})

@app.route("/status/grammar_mcts/<task_id>", methods=["GET"])
def grammar_mcts_status(task_id):
    generations = []
    task = grammar_mcts.AsyncResult(task_id)
    if task.state == "SUCCESS":
        generations = task.get()

    return jsonify({"status": task.state, "generations":generations})