"""
Define a sentence scorer based on BERT model
"""

from typing import *

import numpy as np
import torch
from tqdm.autonotebook import tqdm

from .sentence_score import SentenceScore


class BertScore(SentenceScore):
    """
    Use BERT to score a sentence following the idea describe in the paper
    Effective Sentence Scoring Method Using BERT for Speech Recognition. J. Shin, Y. Lee, Kyomin Jung

    Roughly the idea is to :
    1- mask successively each word of the sentences
        For instance, if the context is "Where is Gael ?" and the sentence to score is "He has left"
        We will create the following mask sentences
            - [CLS] [MASK]  is Gael ?  He has left [SEP]
            - [CLS] Where [MASK]  Gael ? he has left [SEP]
            - ...
            - ...
            - [CLS] Where is Gael ?  he has [MASK] [SEP]

    2- compute the likelihood of each target word that has been mask using context from both side
    3- return the sum all log-likelihood
    """

    def _add_context_and_generate_mask_sentences(self, sentences_token_ids: List[List[int]]) -> List[Dict]:
        full_mask_batch = []
        len_context = len(self.context_ids)

        for sentence_idx, sentence_token_ids in enumerate(sentences_token_ids):
            for token_idx, token in enumerate(sentence_token_ids):
                # construct full sentence : [SEP] context sentence [CLS]
                mask_sentence_token_ids = (
                    [self.tokenizer.cls_token_id]
                    + self.context_ids
                    + sentence_token_ids
                    + [self.tokenizer.sep_token_id]
                )
                # Replace token n°token_idx by [MASK] token
                mask_sentence_token_ids[1 + len_context + token_idx] = self.tokenizer.mask_token_id

                full_mask_batch.append(
                    {
                        "mask_sentence_token_ids": mask_sentence_token_ids,
                        "sentence_idx": sentence_idx,
                        "mask_positions": 1 + len_context + token_idx,
                        "mask_target": token,
                    }
                )

        return full_mask_batch

    def _compute_transformers_log_prob_scores(self, sentences_token_ids: List[List[int]]) -> List[float]:
        """
        1/ First create all the mask_sentences
        2/ Split the mask sentences by batch
            -> The batch can contain mask_sentences coming from different input sentences
            in order to deal with that, the batch will keep for each mask_sentence the following details :
                - mask_sentence_token_ids: list of token ids that compose the mask sentence
                - sentence_idx: index of the corresponding input sentence
                - mask_positions: index of the token that have been masked
                - mask_target: token that have been masked

            those informations allows to :
            1. retrieve the log prob scores of only mask tokens
            2. gather the results by input sentence  at the end
        """
        full_mask_batch = self._add_context_and_generate_mask_sentences(sentences_token_ids)

        mask_log_prob_scores = []
        for i in tqdm(range(0, len(full_mask_batch), self.batch_size), disable=not self.progress_bar):
            batch = full_mask_batch[i : i + self.batch_size]
            mask_log_prob_scores += self._compute_mask_log_prob(batch)

        # Gather the result for each input sentence
        sentences_log_prob_scores = np.zeros(len(sentences_token_ids))
        for mask_sentence_idx, mask_log_prob_score in enumerate(mask_log_prob_scores):
            sentence_idx = full_mask_batch[mask_sentence_idx]["sentence_idx"]
            sentences_log_prob_scores[sentence_idx] += mask_log_prob_score

        return sentences_log_prob_scores.tolist()

    @staticmethod
    def _join_list_of_dict(list_of_dict):
        # for instance, if list_of_dict = [{a:1, b:2}, {a:3, b:4}]
        # will return => {a:[1,3], b:[2,4]}
        return {key: [single_dict[key] for single_dict in list_of_dict] for key in list_of_dict[0].keys()}

    def _compute_mask_log_prob(self, batch: List[Dict]) -> List[float]:
        batch_size = len(batch)
        dict_batch = self._join_list_of_dict(batch)

        input_ids, no_pad_mask = self._pad(
            sequences=list(
                map(lambda ids: torch.tensor(ids, device=self.device), dict_batch["mask_sentence_token_ids"],)
            ),
            pad_token_id=self.tokenizer.sep_token_id,
        )

        with torch.no_grad():
            # contrary to GPT2-based score, we have to provide an attention mask
            # because BERT will also look on the right side and will see the pad tokens
            # with no_pad_mask, the model will zero the score of pad tokens at each layer
            # shape = [batch_size, seq_len, vocab_size]
            logits = self.model(input_ids, attention_mask=no_pad_mask)[0]

            # Retrieve the logits of mask tokens
            # mask_pred_logits.shape = [batch_size, vocac_size]
            mask_pred_logits = logits[range(batch_size), dict_batch["mask_positions"], :]

            # target_score.shape = [batch_size,]
            target_scores = mask_pred_logits[range(batch_size), dict_batch["mask_target"]]
            target_log_probs = target_scores - mask_pred_logits.logsumexp(dim=1)

        return target_log_probs


class BertInverseScore(BertScore):
    """
    Compute P(context | sentence) rather than P(sentence | context)
    """
    def _add_context_and_generate_mask_sentences(self, sentences_token_ids: List[List[int]]) -> List[Dict]:
        full_mask_batch = []

        for sentence_idx, sentence_token_ids in enumerate(sentences_token_ids):
            for token_idx, token in enumerate(self.context_ids):
                # construct full sentence : [SEP] context sentence [CLS]
                mask_sentence_token_ids = (
                    [self.tokenizer.cls_token_id]
                    + self.context_ids
                    + sentence_token_ids
                    + [self.tokenizer.sep_token_id]
                )
                # Replace token n°token_idx by [MASK] token
                mask_sentence_token_ids[1 + token_idx] = self.tokenizer.mask_token_id

                full_mask_batch.append(
                    {
                        "mask_sentence_token_ids": mask_sentence_token_ids,
                        "sentence_idx": sentence_idx,
                        "mask_positions": 1 + token_idx,
                        "mask_target": token,
                    }
                )

        return full_mask_batch