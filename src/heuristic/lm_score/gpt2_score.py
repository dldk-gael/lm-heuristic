from heuristic.score import Score
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import math


class GPT2Score(Score):
    """
    Compute the perplexity score of a sentence given a langage model :
     - average the loglikelihood that the LM assign to each token of the sentence given the previous tokens
    """
    def __init__(self, model_name, batch_size, length_normalization=False):
        """
        Initialize the pre-trained GPT2 model
        :param model_name : "gpt2", "gpt2-medium" or "gpt2-large"
        :param length_normalization [boolean]
        :param batch_size: int
        """
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.length_normalization = length_normalization
        self.batch_size = batch_size

    def compute_score(self, sentences):
        """
        Compute GPT2 score of the sentences
        :param sentences: str | List[str] sentences to evaluate
        :return: List[float], list of score
        """
        sentences = [sentences] if type(sentences) == str else sentences

        def encode_with_bos_eos_tokens(text):
            return self.tokenizer.encode(self.tokenizer.bos_token + text + self.tokenizer.eos_token)

        # Prepare the input ids
        tokens_ids = [encode_with_bos_eos_tokens(sentence) for sentence in sentences]

        input_ids = pad_sequence(list(map(torch.tensor, tokens_ids)), batch_first=True)

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            input_ids = input_ids.cuda()

        # Compute all prediction logits by batch of batch_size
        with torch.no_grad():
            i = 0
            pred_scores = []
            while i + self.batch_size < input_ids.shape[0]:
                pred_scores.append(self.model(input_ids[i: i + self.batch_size])[0])
                i += self.batch_size
            if i < input_ids.shape[0]:
                pred_scores.append(self.model(input_ids[i:])[0])

        # pred_scores.shape = [len(sentences), max(len(tokenize(sentence)) + 2), vocab_size]
        pred_scores = torch.cat(pred_scores, dim=0)
        with torch.no_grad():
            log_scores = torch.nn.LogSoftmax(dim=2)(pred_scores)
        log_scores = log_scores[:, :-1, :]  # To skip prediction for EOS tokens

        sentences_score = []
        for i in range(len(sentences)):
            sentence_token_ids = tokens_ids[i][1:]  # len = nb token in sentence + 1 (for BOS)
            sentence_token_scores = log_scores[i, range(len(sentence_token_ids)), sentence_token_ids]
            if self.length_normalization:
                sentences_score.append(math.exp(torch.mean(sentence_token_scores).item()))
            else:
                sentences_score.append(math.exp(torch.sum(sentence_token_scores).item()))

        return sentences_score
