from flask import request, jsonify, Blueprint
from grammar_backend import app, paraphrase_generator, random_searcher, montecarlo_searcher
from lm_heuristic.generation import generate_from_cfg
from lm_heuristic.tree import CFGrammarNode

views = Blueprint('views', __name__)

@app.route("/ping", methods=["GET"])
def ping():
    print("pong")
    return "pong"

@app.route("/paraphrase", methods=["POST"])
def paraphrase():
    data = request.get_json()
    paraphrases = paraphrase_generator(
        sentence=data["sentence_to_paraphrase"],
        forbidden_words=data["forbidden_words"],
        nb_samples=data["number_of_samples"],
        top_n_to_keep=data["keep_top"],
    )
    return jsonify({"paraphrases": paraphrases})

@app.route("/grammar_sampling", methods=["POST"])
def grammar_sampling():
    print("here")
    data = request.get_json()
    grammar_root = CFGrammarNode.from_string(data["grammar"])
    if data["strategy"] == "MCTS":
        generations = generate_from_cfg(
            grammar_root, montecarlo_searcher, data["number_of_samples"], data["keep_top"]
        )
    elif data["strategy"] == "Random Sampling":
        generations = generate_from_cfg(
            grammar_root, random_searcher, data["number_of_samples"], data["number_of_samples"]
        )
    return jsonify({"generations": generations})    

