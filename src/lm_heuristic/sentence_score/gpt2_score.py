"""
Define a sentence scorer based on GPT2 model
"""

from typing import *
import logging

import torch
from tqdm.autonotebook import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

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

    def _build(self):
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        if not self.model:
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name)
        self.model.eval()
        self.model.to(self.device)

    def _compute_sentences_scores(self, sentences: List[str]) -> List[float]:
        scores = []
        for i in tqdm(range(0, len(sentences), self.batch_size), disable=not self.progress_bar):
            batch = sentences[i : i + self.batch_size]
            scores += self._compute_single_batch(batch)

        if len(sentences) % self.batch_size != 0:
            last_batch = sentences[-(len(sentences) % self.batch_size) :]
            scores += self._compute_single_batch(last_batch)

        return scores

    def _pad(self, sequences: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        rewrite torch.nn.utils.rnn.pad_sequence so that it return a boolean mask of pad position
        the advantage is that we can avoid to add custom pad token to the model and pad directly
        with eos token
        """
        max_seq_len = max([s.size(0) for s in sequences])
        out_tensor = (
            sequences[0].data.new(len(sequences), max_seq_len).fill_(self.tokenizer.eos_token_id)  # type:ignore
        )
        mask = torch.zeros((len(sequences), max_seq_len), device=sequences[0].device)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1

        return out_tensor, mask

    def _add_bos_token_and_encode(self, text: str) -> List[float]:
        # TODO encode the sentences by batch and not one by one 
        # can be much faster when using FastTokenizer (because of // that occurs in the back)
        return self.tokenizer.encode(self.tokenizer.bos_token + text)

    def _compute_single_batch(self, sentences: List[str]) -> List[float]:
        # Prepare the input ids
        tokens_ids = [self._add_bos_token_and_encode(sentence) for sentence in sentences]
        # don't count the bos token
        sentences_len = torch.tensor(  # pylint: disable=not-callable
            [len(toks) - 1 for toks in tokens_ids], device=self.device
        )
        input_ids, mask = self._pad(list(map(torch.tensor, tokens_ids)))

        input_ids = input_ids.to(self.device)
        mask = mask.to(self.device)

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
            tokens_scores *= mask[:, 1:]

            sentences_score = torch.sum(tokens_scores, dim=1)
            if self.length_normalization:
                sentences_score = sentences_score / sentences_len

            sentences_score = torch.exp(sentences_score).tolist()  # type: ignore
        return sentences_score  # type: ignore
