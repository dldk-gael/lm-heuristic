import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_login import LoginManager
from celery import Celery

from lm_heuristic.tree_search import RandomSearch, MonteCarloTreeSearch
from lm_heuristic.tree import CFGrammarNode
from lm_heuristic.sentence_score import GPT2Score
from lm_heuristic.heuristic import Heuristic
from lm_heuristic.generation import GPT2Paraphrases, generate_from_cfg


def run_flask_server(args):
    # LOAD IN MEMORY LANGUAGE MODEL 

    with open(args.paraphrase_context, "r") as file:
        paraphrase_context = file.read()

    paraphrase_generator = GPT2Paraphrases(
        args.paraphrase_LM,
        paraphasing_context=paraphrase_context,
        question_paraphrasing=False,
        batch_size=args.batch_size,
    )

    no_heuristic = Heuristic(lambda terminal_nodes: [0] * len(terminal_nodes))
    random_searcher = RandomSearch(no_heuristic)
    montecarlo_searcher = MonteCarloTreeSearch(no_heuristic)

    # LAUNCH FLASK SERVER 
    app = Flask(__name__)
    CORS(app)
    user_id = 0

    #app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
    #app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
    #celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
    #celery.conf.update(app.config)

    @app.route("/", methods=["GET"])
    def handle_get():
        return "pong"

    @app.route("/", methods=["POST"])
    def handle_post():
        print(request.url)
        data = request.get_json()
        print(data)
        if data["order"] == "grammar_sample":
            return jsonify({"generations": generate_from_grammar(data)})

        if data["order"] == "paraphrase":
            return jsonify({"paraphrases": paraphrase(data)})

    def generate_from_grammar(data):
        grammar_root = CFGrammarNode.from_string(data["grammar"])
        if data["strategy"] == "MCTS":
            return generate_from_cfg(grammar_root, montecarlo_searcher, data["number_of_samples"], data["keep_top"])
        elif data['strategy'] == "Random Sampling":
            return generate_from_cfg(
                grammar_root, random_searcher, data["number_of_samples"], data["number_of_samples"]
            )
    
    def paraphrase(data):
        return paraphrase_generator(
            sentence=data["sentence_to_paraphrase"],
            forbidden_words=data["forbidden_words"],
            nb_samples=data["number_of_samples"],
            top_n_to_keep=data["keep_top"],
        )

    app.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=1, type=int, help="Specify the batch size for GPT2 input")
    parser.add_argument(
        "--paraphrase_LM",
        default="gpt2",
        type=str,
        help="LM to use for paraphrasing : gpt2, gpt2-medium or gpt2-large",
    )
    parser.add_argument(
        "--paraphrase_context",
        default="data/text/paraphrase.txt",
        type=str,
        help="Text to use for paraphrase context",
    )

    args = parser.parse_args()
    run_flask_server(args)
