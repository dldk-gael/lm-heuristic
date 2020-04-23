from heuristic.score import Score
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from tqdm.autonotebook import tqdm


class GPT2Score(Score):
    """
    Compute the perplexity score of a sentence given a langage model :
     - average the loglikelihood that the LM assign to each token of the sentence given the previous tokens
    """
    def __init__(self, model_name, batch_size=1, length_normalization=False, verbose=False):
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
        self.verbose = verbose

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
        input_ids = pad_sequence(list(map(torch.tensor, tokens_ids)), batch_first=True)

        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # Compute all prediction logits by batch of batch_size
        with torch.no_grad():
            pred_logits = self.model(input_ids)[0]

        # pred_scores.shape = [batch_size, seq_len, vocab_size]
        pred_logits = pred_logits[:, :-1, :]  # To skip prediction for last tokens
        pred_scores = torch.nn.LogSoftmax(dim=2)(pred_logits)

        scores = []
        # TODO: look for a trick to skip this loop
        for i in range(len(sentences)):
            sentence_token_ids = tokens_ids[i][1:]  # To shift 1-rigth the gold label
            sentence_token_scores = pred_scores[i, range(len(sentence_token_ids)), sentence_token_ids]
            if self.length_normalization:
                scores.append(torch.exp(torch.mean(sentence_token_scores)).item())
            else:
                scores.append(torch.exp(torch.sum(sentence_token_scores)).item())

        return scores
