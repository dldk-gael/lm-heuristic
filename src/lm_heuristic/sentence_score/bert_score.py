"""
Define a sentence scorer based on BERT model
"""

from typing import *

from transformers import BertForMaskedLM
import torch
from tqdm.autonotebook import tqdm

from .sentence_score import SentenceScore


class BertScore(SentenceScore):
    """
    Use BERT to score a sentence following the idea describe in the paper
    Effective Sentence Scoring Method Using BERT for Speech Recognition. J. Shin, Y. Lee, Kyomin Jung

    Roughly the idea is to :
    1- mask successively each word of the sentences
        For instance, if the context is "Where is Gael ?" and a possible sentence is "He has left"
        We will create the following mask sentences
            - [CLS] [MASK]  is Gael ?  He has left [SEP]
            - [CLS] Where [MASK]  Gael ? he has left [SEP]
            - ...
            - ...
            - [CLS] Where is Gael ?  he has [MASK] [SEP]

    2- compute the likelihood of each target word that has been mask using context from both side
    3- return the sum or average of all log-likelihood
    """

    def build(self):
        self.model = BertForMaskedLM.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def _compute_sentences_scores(self, sentences: List[str]) -> List[float]:
        # TODO merge the PR I made on LM_scorer to batch by sentences
        return [
            self.compute_single_sentence_score(sentence) for sentence in tqdm(sentences, disable=not self.progress_bar)
        ]

    def compute_single_sentence_score(self, sentence: str) -> float:
        """
        Compute BERT score of a sentence
        :param sentence
        :return: float, score
        """
        # TODO merge the PR I made on LM_scorer to speed up BERT scoring

        # prepare the batch of mask sentences
        tok_sentence = self.tokenizer.tokenize(sentence)
        encoded_sentence = self.tokenizer.convert_tokens_to_ids(tok_sentence)
        seq_len = len(tok_sentence)

        mask_sentences = [tok_sentence.copy() for _ in range(seq_len)]
        for i in range(seq_len):
            mask_sentences[i][i] = "[MASK]"

        input_sentences = [["[CLS]"] + mask_sentence + ["[SEP]"] for mask_sentence in mask_sentences]
        input_ids = torch.stack(
            [
                torch.tensor(self.tokenizer.convert_tokens_to_ids(input_sentence))  # pylint: disable=not-callable
                for input_sentence in input_sentences
            ],
            dim=0,
        )
        input_ids = input_ids.to(self.device)

        # Compute all prediction logits by batch
        i = 0
        pred_logits = []
        with torch.no_grad():
            while i + self.batch_size < seq_len:
                pred_logits.append(self.model(input_ids[i : i + self.batch_size])[0])
                i += self.batch_size
            if i < seq_len:
                pred_logits.append(self.model(input_ids[i:])[0])

        all_pred_logits = torch.cat(pred_logits, dim=0)  # shape (seq_len, seq_len, vocab_size)

        # retrieve only logits corresponding to mask tokens : new shape (seq_len, vocab_size)
        mask_positions = range(1, 1 + seq_len)  # not take into account first and last special tokens
        mask_pred_logits = all_pred_logits[range(input_ids.shape[0]), mask_positions, :]

        # compute log_likelihood and retrieve value for true_tokens
        log_likelihood_scores = torch.nn.LogSoftmax(dim=1)(mask_pred_logits)
        log_likelihood_scores = log_likelihood_scores[range(input_ids.shape[0]), encoded_sentence]

        if self.length_normalization:
            return torch.exp(torch.mean(log_likelihood_scores)).item()
        else:
            return torch.exp(torch.sum(log_likelihood_scores)).item()
