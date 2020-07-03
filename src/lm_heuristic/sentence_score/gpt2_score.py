from typing import *  # pylint: disable=unused-wildcard-import
import logging

import torch
from tqdm.autonotebook import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel

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

    def __init__(
        self,
        model_name: str,
        model: GPT2LMHeadModel = None,
        batch_size: int = 1,
        length_normalization: bool = False,
        device: str = None,
        verbose: bool = False,
    ):
        SentenceScore.__init__(
            self,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            length_normalization=length_normalization,
            verbose=verbose,
        )

        if model is not None:
            logger.info("Loading %s on %s", model_name, self.device)
            self.model = model
            self.model.eval()
            self.model.to(self.device)

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def build(self):
        if not self.model:
            logger.info("Loading %s on %s", self.model_name, self.device)
            self.model = GPT2LMHeadModel.from_pretrained(self.model_name).to(self.device)

    def _compute_sentences_scores(self, sentences: List[str]) -> List[float]:
        """
        :param text: str | List[str] sentences to evaluate
        :return list of sentence's score
        """
        scores = []
        for i in tqdm(range(0, len(sentences), self.batch_size), disable=not self.verbose):
            batch = sentences[i : i + self.batch_size]
            scores += self._compute_single_batch(batch)

        if len(sentences) % self.batch_size != 0:
            last_batch = sentences[-(len(sentences) % self.batch_size) :]
            scores += self._compute_single_batch(last_batch)

        return scores

    def _pad(self, sequences: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param sequences: list of Tensor
        :return: padding input + mask,
                  both are rensors of shape (len(sequences), max sequence length)
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
        return self.tokenizer.encode(self.tokenizer.bos_token + text)

    def _compute_single_batch(self, sentences: List[str]) -> List[float]:
        """
        Compute GPT2 score of the sentences by given the model all the sentences in a single batch
        :param sentences: str | List[str] sentences to evaluate:
        :return: List[float], list of score
        """
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
        # (found this nice trick in lm-scorer package source code)
        tokens_scores = pred_scores.gather(dim=2, index=target_ids.unsqueeze(2)).squeeze(2)

        # Zeros the score of pad tokens
        tokens_scores *= mask[:, 1:]

        sentences_score = torch.sum(tokens_scores, dim=1)
        if self.length_normalization:
            sentences_score = sentences_score / sentences_len

        sentences_score = torch.exp(sentences_score).tolist()  # type: ignore
        return sentences_score  # type: ignore
