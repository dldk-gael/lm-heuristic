"""
Define an abstract class from which all transformers-based sentence scorer must inherate
"""

from abc import ABC, abstractmethod
from typing import *
import logging

from transformers import PreTrainedTokenizer, PreTrainedModel
import torch


logger = logging.getLogger(__name__)


class SentenceScore(ABC):
    """
    A SentenceScore object compute a "naturalness" score for a sentence or a list of sentences
    from the output of a transformer models (BERT, GPT2, ...).

    Typically, given a sentence, the scorer will compute a score for each sentence's tokens and
    then return the average as a sentence score
    """

    def __init__(
        self,
        model_name: str = "",
        model: PreTrainedModel = None,
        batch_size: int = 1,
        length_normalization: bool = False,
        device: str = None,
        progress_bar: bool = False,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.length_normalization = length_normalization
        self.progress_bar = progress_bar
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # The transformers models will only be load in memory when using build methods
        # This allows to avoid surcharging the memory when you do not want to use the scorer directly
        # An already loaded-in-memory model can also be passed to the scorer
        self.model = model
        self.is_already_built = False

        self.context = ""
        self.context_ids: List[int] = []
        self.tokenizer: PreTrainedTokenizer

    def build(self):
        self._build()
        self.is_already_built = True
        return self
        
    @abstractmethod
    def set_context(self, context):
        """
        The context will be concatenate at the left side of input sentence
        before sentence evaluation
        """
        ...

    @abstractmethod
    def _build(self):
        ...
        
    @overload
    def compute_score(self, text: str) -> float:
        ...

    @overload
    def compute_score(self, text: List[str]) -> List[float]:
        ...

    def compute_score(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        if not self.is_already_built:
            self.build()

        sentences = [text] if isinstance(text, str) else text
        scores = self._compute_sentences_scores(sentences)
        return scores[0] if isinstance(text, str) else scores

    def __call__(self, sentences):
        return self.compute_score(sentences)

    @abstractmethod
    def _compute_sentences_scores(self, sentences: List[str]) -> List[float]:
        ...

    # /////////////////////////////////////////////////////////////////
    # Define some methods to make it easier to test the sentence scorer
    # /////////////////////////////////////////////////////////////////

    def rank_sentences(self, sentences: List[str]) -> List[Tuple[str, float]]:
        """
        Rank the sentences by their scores
        :param sentences: list[str]
        :return: sorted list of sentence with their associated scores
        """
        scored_sentences = list(zip(sentences, self.compute_score(sentences)))
        return sorted(scored_sentences, key=lambda x: x[1], reverse=True)

    def print_sentences_score(self, sentences: List[str]):
        """
        Print each sentence from the most probable to the less probable along with their scores
        """
        self.print_ranked_sentences(self.rank_sentences(sentences))

    @staticmethod
    def print_ranked_sentences(ranked_sentences, nb_to_print=-1):
        """
        Print the nb_to_print best scored sentences
        """
        for i, (sentence, score_result) in enumerate(ranked_sentences):
            print("\tn°%d (score : %f) - %s" % (i + 1, score_result, sentence))
            if i == nb_to_print:
                break
        print("")
