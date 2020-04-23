from heuristic.score import Score
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm.autonotebook import tqdm


class GPT2Score(Score):
    """
    Compute the score of a sentence for GPT2 model.
    Because GPT2 has been trained to predict next_tokens given all previous tokens, we use as a score :
    P(sentence) = P(t_n | t_1 .. t_(n-1)) * ... * P(t_1)
    Computation are performed in logspace

    Length normalization can be applied on top of this score:
    ->  average the loglikelihood of each token
    """
    def __init__(self, model_name, batch_size=1, length_normalization=False):
        """
        Initialize the pre-trained GPT2 model
        :param model_name : "gpt2", "gpt2-medium" or "gpt2-large"
        :param length_normalization [boolean]
        :param batch_size: int
        :param verbose: [Boolean]
        """
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.length_normalization = length_normalization
        self.batch_size = batch_size

    def compute_score(self, sentences):
        scores = []
        for i in tqdm(range(len(sentences) // self.batch_size)):
            scores += self.compute_single_batch(sentences[i * self.batch_size: (i+1) * self.batch_size])
        if len(sentences) % self.batch_size != 0:
            scores += self.compute_single_batch(sentences[-len(sentences) % self.batch_size:])
        return scores

    def compute_single_batch(self, sentences):
        """
        Compute GPT2 score of the sentences
        :param sentences: str | List[str] sentences to evaluate:
        :return: List[float], list of score
        """
        sentences = [sentences] if type(sentences) == str else sentences

        def encode_with_bos_eos_tokens(text):
            # TODO Evaluate if it works best when using eos_token
            # TODO Same question for bos_token
            return self.tokenizer.encode(self.tokenizer.bos_token + text)

        # Prepare the input ids
        tokens_ids = [encode_with_bos_eos_tokens(sentence) for sentence in sentences]
        sentences_len = torch.tensor([len(toks) - 1 for toks in tokens_ids])  # do not count the bos token
        input_ids = pad_sequence(list(map(torch.tensor, tokens_ids)),
                                 batch_first=True,
                                 padding_value=0)

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # Compute all prediction logits by batch of batch_size
        with torch.no_grad():
            pred_logits = self.model(input_ids)[0]  # shape = [batch_size, seq_len, vocab_size]

        pred_scores = torch.nn.LogSoftmax(dim=2)(pred_logits)

        # Align input and target
        target_ids = input_ids[:, 1:]
        pred_scores = pred_scores[:, :-1, :]

        tokens_scores = pred_scores.gather(dim=2, index=target_ids.unsqueeze(2)).squeeze(2)

        # Zeros the score of pad tokens
        tokens_scores = torch.where(input_ids[:, :-1] != -1, tokens_scores, torch.zeros(tokens_scores.shape))
        sentences_score = torch.sum(tokens_scores, dim=1)
        if self.length_normalization:
            sentences_score = sentences_score / sentences_len

        sentences_score = torch.exp(sentences_score).tolist()
        return sentences_score

