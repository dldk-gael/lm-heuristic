import os

from transformers import GPT2LMHeadModel
import tensorflow_hub as hub
from nltk.parse import CoreNLPParser

from lm_heuristic.generation import GPT2Paraphrases
from lm_heuristic.sentence_score import GPT2Score
from lm_heuristic.tree_search.evaluator import Evaluator
from lm_heuristic.tree_search.mcts import MonteCarloTreeSearch, AllocationStrategy
from lm_heuristic.tree_search.random import RandomSearch
from lm_heuristic.tree.interface.nltk_grammar import FeatureGrammarNode
from lm_heuristic.utils.zero_scorer import ZeroScorer

from .config import config


class Models:
    """
    This class is used to load in memory the different language model only when
    it is necessary.
    """

    def __init__(self):
        self._gpt2_model = None
        self._universal_sentence_encoder = None
        self.paraphrase_generator = None
        self.montecarlo_searcher = None
        self.random_searcher = None
        self.parser = None

    def is_paraphrase_generator_ready(self):
        return self.paraphrase_generator is not None

    def load_paraphrase_generator(self):
        assert not self.is_paraphrase_generator_ready()
        with open(config["PATH_TO_PARAPHRASE_CONTEXT"], "r") as file:
            paraphrase_context = file.read()

        self.paraphrase_generator = GPT2Paraphrases(
            gpt2_model_name=config["GPT2_NAME"],
            gpt2_model=self.gpt2(),
            paraphasing_context=paraphrase_context,
            question_paraphrasing=False,
            sentence_encoder=self.universal_sentence_encoder(),
            batch_size=config["BATCH_SIZE"],
        )

    def is_montecarlo_searcher_ready(self):
        return self.montecarlo_searcher is not None

    def load_montecarlo_searcher(self):
        assert not self.is_montecarlo_searcher_ready()
        gpt_2_scorer = GPT2Score(
            model_name=config["GPT2_NAME"],
            model=self.gpt2(),
            batch_size=config["BATCH_SIZE"],
            length_normalization=True,
        )

        evaluator = Evaluator(gpt_2_scorer)
        evaluator.set_default_value(node=FeatureGrammarNode("DEAD_END", feature_grammar=None), value=0.0)

        self.montecarlo_searcher = MonteCarloTreeSearch(
            evaluator=evaluator, buffer_size=config["BATCH_SIZE"], progress_bar=False,
        )

    def is_random_searcher_ready(self):
        return self.random_searcher is not None

    def load_random_searcher(self):
        assert not self.is_random_searcher_ready()
        self.random_searcher = RandomSearch(Evaluator(ZeroScorer()))

    def gpt2(self):
        if not self._gpt2_model:
            self._gpt2_model = GPT2LMHeadModel.from_pretrained(config["GPT2_NAME"])
        return self._gpt2_model

    def universal_sentence_encoder(self):
        if not self._universal_sentence_encoder:
            assert os.path.exists(config["PATH_TO_UNIVERSAL_SENTENCE_ENCODER"]), "USE weights was not found."
            self._universal_sentence_encoder = hub.load(config["PATH_TO_UNIVERSAL_SENTENCE_ENCODER"])
        return self._universal_sentence_encoder

    def is_parser_ready(self):
        return self.parser is not None

    def load_parser(self):
        assert not self.is_parser_ready()
        self.parser = CoreNLPParser(url="http://localhost:9000")
