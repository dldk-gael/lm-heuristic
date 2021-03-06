"""
Define an abstract class from which all transformers-based sentence scorer must inherate
"""

from abc import ABC, abstractmethod
from typing import *
import logging
import numpy as np
import math

from transformers import AutoModelWithLMHead, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, BatchEncoding

import torch

from .unigram import load_unigram

logger = logging.getLogger(__name__)


class SentenceScore(ABC):
    """
    A SentenceScore object compute a "naturalness" score for a sentence or a list of sentences
    from the output of a transformer models (BERT, GPT2, ...).

    Typically, given a sentence, the scorer will compute a score for each sentence's tokens and
    then perform some reductions to get a sentence score. Possible reduction technics have been
    studied by in Jey Han Lau, Carlos Armendariz, Shalom Lappin, Matthew Purver, and Chang Shu
    and presented in the paper (2020) "How Furiously Can Colorless Green Ideas Sleep? Sentence
    Acceptability in Context."
    """

    def __init__(
        self,
        model_name: str = "",
        model: PreTrainedModel = None,
        batch_size: int = 1,
        device: str = None,
        progress_bar: bool = False,
        load_unigram_file: bool = False,
        normalization_strategy="LP",
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.progress_bar = progress_bar
        self.device = device if device else ("cuda:0" if torch.cuda.is_available() else "cpu")

        # The transformers models will only be load in memory when using build methods
        # This allows to avoid surcharging the memory when you do not want to use the scorer directly
        # An already loaded-in-memory model can also be passed to the scorer
        self.model = model
        self.is_already_built = False

        self.context = None
        self.context_ids: List[int] = []
        self.tokenizer: PreTrainedTokenizer

        self.load_unigram_file = load_unigram_file
        if self.load_unigram_file:
            self.unigram_count = load_unigram(model_name)
            self.unigram_total = sum(self.unigram_count.values())

        self.normalization_strategy = normalization_strategy

    def build(self):
        if self.is_already_built:
            return self

        self.is_already_built = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        self.model = AutoModelWithLMHead.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        return self

    def set_context(self, context):
        self.context = context
        self.context_ids = self.tokenizer(context, add_special_tokens=False)["input_ids"] if self.context else []

    @abstractmethod
    def _compute_transformers_log_prob_scores(self, sentences_token_ids: List[List[int]]) -> List[float]:
        """
        Given a list of tokenized and encoded sentences
        return the list of log probability of each sentences for the Language Model
        """
        ...

    def _compute_unigram_log_prob(self, tokens: List[str]) -> float:
        assert (
            self.load_unigram_file
        ), "Try to compute the unigram log prob of a sentence but no unigram count pickle file was loaded"
        count = np.array([self.unigram_count[token] for token in tokens if token not in ["Ċ", "Âł"]]) # remove '\n' 

        return np.sum(np.log(count / self.unigram_total))

    def score_normalization(self, sentence_score: float, tokens: List[str]):
        if self.normalization_strategy == "LP":
            return sentence_score
        elif self.normalization_strategy == "MeanLP":
            return sentence_score / len(tokens)
        elif self.normalization_strategy == "PenLP":
            return sentence_score / ((5 + len(tokens)) / (5 + 1)) ** 0.8
        elif self.normalization_strategy == "NormLP":
            return -sentence_score / self._compute_unigram_log_prob(tokens)
        elif self.normalization_strategy == "SLOR":
            return (sentence_score - self._compute_unigram_log_prob(tokens)) / len(tokens)

        raise NotImplementedError(
            """Only the following strategies are implemeted : \n
            LP, MeanLP, PenLP, NormLP, SLOR"""
        )

        
    def compute_score(self, text: Union[str, List[str]]) -> Union[float, List[float]]:

        assert self.is_already_built, "You have to first build the model."

        sentences = [text] if isinstance(text, str) else text

        if self.context_ids != []:
            # Because in BPE, tokenisation is different if there is a space before a word
            sentences = [" " + sentence for sentence in sentences]

        # We can not directly input the special tokens because we first have to insert the context
        encoding: BatchEncoding = self.tokenizer(sentences, add_special_tokens=False)
        raw_sentences_score = self._compute_transformers_log_prob_scores(encoding["input_ids"])

        normalized_sentences_scores = []
        for i, sentence_score in enumerate(raw_sentences_score):
            normalized_sentences_scores.append(math.exp(self.score_normalization(sentence_score, encoding.tokens(i))))

        return normalized_sentences_scores[0] if isinstance(text, str) else normalized_sentences_scores

    def __call__(self, sentences):
        return self.compute_score(sentences)

    @staticmethod
    def _pad(sequences: List[torch.Tensor], pad_token_id) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        rewrite torch.nn.utils.rnn.pad_sequence so that it return a boolean mask of pad position
        the advantage is that we can avoid to add custom pad token to the model and pad directly
        with any token we want
        """
        max_seq_len = max([s.size(0) for s in sequences])
        out_tensor = sequences[0].data.new(len(sequences), max_seq_len).fill_(pad_token_id)  # type:ignore
        mask = torch.zeros((len(sequences), max_seq_len), device=sequences[0].device)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1

        return out_tensor, mask

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
