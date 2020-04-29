from heuristic.score import Score
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm.autonotebook import tqdm
import logging

logger = logging.getLogger()


class GPT2Score(Score):
    """
    Compute the score of a sentence for GPT2 model.
    Because GPT2 has been trained to predict next_tokens given all previous tokens, we use as a score :
    P(sentence) = P(t_n | t_1 .. t_(n-1)) * ... * P(t_1)
    Computation are performed in logspace

    Length normalization can be applied on top of this score:
    ->  average the loglikelihood of each token
    """
    def __init__(self, model_name, batch_size=1, length_normalization=False, verbose=False):
        """
        Initialize the pre-trained GPT2 model
        :param model_name : "gpt2", "gpt2-medium" or "gpt2-large"
        :param length_normalization [boolean]
        :param batch_size: int
        :param verbose: bool, if true will print a tqdm progress bar during computation
        """
        Score.__init__(self)
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        logger.info("Loading %s on %s" % (model_name, self.device))
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.length_normalization = length_normalization
        self.batch_size = batch_size
        self.verbose = verbose

    def compute_score(self, text):
        """
        :param text: str | List[str] sentences to evaluate
        :return list of sentence's score
        """
        sentences = [text] if type(text) == str else text

        scores = []
        for i in tqdm(range(len(sentences) // self.batch_size), disable=not self.verbose):
            scores += self.compute_single_batch(sentences[i * self.batch_size: (i+1) * self.batch_size])
        if len(sentences) % self.batch_size != 0:
            scores += self.compute_single_batch(sentences[- (len(sentences) % self.batch_size):])

        return scores[0] if type(text) == str else scores

    def pad(self, sequences):
        """
        :param sequences: list of Tensor
        :return: padding input + mask,
                  both are rensors of shape (len(sequences), max sequence length)
        """
        max_seq_len = max([s.size(0) for s in sequences])
        out_tensor = sequences[0].data.new(len(sequences), max_seq_len).fill_(self.tokenizer.eos_token_id)
        mask = torch.zeros((len(sequences), max_seq_len), device=sequences[0].device)
        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length] = tensor
            mask[i, :length] = 1

        return out_tensor, mask

    def add_bos_token_and_encode(self, text):
        return self.tokenizer.encode(self.tokenizer.bos_token + text)

    def compute_single_batch(self, sentences):
        """
        Compute GPT2 score of the sentences by given the model all the sentences in a single batch
        :param sentences: str | List[str] sentences to evaluate:
        :return: List[float], list of score
        """
        # Prepare the input ids
        tokens_ids = [self.add_bos_token_and_encode(sentence) for sentence in sentences]
        # don't count the bos token
        sentences_len = torch.tensor([len(toks) - 1 for toks in tokens_ids], device=self.device)
        input_ids, mask = self.pad(list(map(torch.tensor, tokens_ids)))

        input_ids = input_ids.to(self.device)
        mask = mask.to(self.device)

        # Compute all prediction logits by batch of batch_size
        with torch.no_grad():
            pred_logits = self.model(input_ids)[0]  # shape = [batch_size, seq_len, vocab_size]

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

        sentences_score = torch.exp(sentences_score).tolist()
        return sentences_score

