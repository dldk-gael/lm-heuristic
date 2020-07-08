"""
Define a sentence scorer based on GPT2 model
"""

from typing import *
import logging

import torch
from tqdm.autonotebook import tqdm

from .sentence_score import SentenceScore


logger = logging.getLogger()


class GPT2Score(SentenceScore):
    """
    Compute the score of a sentence for GPT2 model.
    Because GPT2 has been trained to predict next_tokens given all previous tokens, we use as a score :
    P(sentence) = P(t_n | t_1 .. t_(n-1)) * ... * P(t_1)
    Computation are performed in logspace

    Length normalization can be applied on top of this score:
    ->  average the loglikelihood of each token
    """

    def _compute_LM_log_prob_scores(self, sentences_token_ids: List[List[int]]) -> List[float]:
        log_prob_scores = []

        for i in tqdm(range(0, len(sentences_token_ids), self.batch_size), disable=not self.progress_bar):
            batch = sentences_token_ids[i : i + self.batch_size]
            log_prob_scores += self._compute_single_batch(batch)

        return log_prob_scores

    def _compute_single_batch(self, sentences_token_ids: List[List[int]]) -> List[float]:
        # Prepare the input ids
        tokens_ids = [
            [self.tokenizer.bos_token_id] + self.context_ids + sentence_token_ids
            for sentence_token_ids in sentences_token_ids
        ]

        input_ids, no_pad_mask = self._pad(
            sequences=list(map(lambda ids: torch.tensor(ids, device=self.device), tokens_ids)),
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Compute all prediction logits by batch of batch_size
        with torch.no_grad():
            # shape = [batch_size, seq_len, vocab_size]
            pred_logits = self.model(input_ids)[0]
            pred_scores = torch.nn.LogSoftmax(dim=2)(pred_logits)

            # Align input and target
            target_ids = input_ids[:, 1:]
            pred_scores = pred_scores[:, :-1, :]

            # Retrieve the token scores corresponding to the target id
            tokens_scores = pred_scores.gather(dim=2, index=target_ids.unsqueeze(2)).squeeze(2)

            # Zeros the score of pad tokens
            tokens_scores *= no_pad_mask[:, 1:]

        return torch.sum(tokens_scores, dim=1).tolist()
