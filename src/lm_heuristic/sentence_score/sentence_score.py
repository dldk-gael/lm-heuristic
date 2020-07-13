"""
Define an abstract class from which all transformers-based sentence scorer must inherate
"""

from abc import ABC, abstractmethod
from typing import *
import logging
import pickle
import os
import numpy as np
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    BatchEncoding,
    PreTrainedTokenizerFast,
    PreTrainedModel,
)
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
        load_unigram_file: bool = True,
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
        self.tokenizer: PreTrainedTokenizerFast

        self.load_unigram_file = load_unigram_file
        if self.load_unigram_file:
            self.unigram_count = load_unigram(model_name)
            self.unigram_total = sum(self.unigram_count.values())

    def build(self):
        self.is_already_built = True

        # TODO this can be replace by AutoTokenizer and AutoModelWithLMHead
        if "gpt" in self.model_name.lower():
            self.tokenizer = GPT2TokenizerFast.from_pretrained(self.model_name)
            if not self.model:
                self.model = GPT2LMHeadModel.from_pretrained(self.model_name)

        elif "bert" in self.model_name.lower():
            self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)
            if not self.model:
                self.model = BertForMaskedLM.from_pretrained(self.model_name)

        else:
            raise NotImplementedError("Sentence scorer only work with gpt2-based and BERT-based model")

        self.model.to(self.device)
        self.model.eval()

    def set_context(self, context):
        self.context = context
        self.context_ids = self.tokenizer(context)["input_ids"] if self.context else []

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
        count = np.array([self.unigram_count[token] for token in tokens])
        return np.sum(np.log(count / self.unigram_total))

    def score_normalization(self, sentence_score: float, tokens: List[str], normalization_strategy="raw_log_prob"):
        if normalization_strategy == "raw_log_prob":
            return sentence_score
        elif normalization_strategy == "mean_length_log_prob":
            return sentence_score / len(tokens)
        elif normalization_strategy == "mean_length_alpha_log_prob":
            return sentence_score / ((5 + len(tokens)) / (5 + 1)) ** 0.8
        elif normalization_strategy == "unigram_norm_log_prob":
            return -sentence_score / self._compute_unigram_log_prob(tokens)
        raise NotImplementedError(
            """Only the following strategies are implemeted : \n
            raw_log_prob, mean_length_log_prob, mean_length_alpha_log_prob, unigram_norm_log_prob"""
        )

    def compute_score(
        self, text: Union[str, List[str]], context: str = None, normalization_strategy: str = "raw_log_prob"
    ) -> Union[float, List[float]]:

        if not self.is_already_built:
            self.build()

        if context:
            self.set_context(context)

        sentences = [text] if isinstance(text, str) else text

        if self.context_ids != []:
            # Because in BPE, tokenisation is different if there is a space before a word
            sentences = [" " + sentence for sentence in sentences]

        # We can not directly input the special tokens because we first have to insert the context
        encoding: BatchEncoding = self.tokenizer(sentences, add_special_tokens=False)
        raw_sentences_score = self._compute_transformers_log_prob_scores(encoding["input_ids"])

        normalized_sentences_scores = []
        for i, sentence_score in enumerate(raw_sentences_score):
            normalized_sentences_scores.append(
                self.score_normalization(sentence_score, encoding.tokens(i), normalization_strategy)
            )

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
