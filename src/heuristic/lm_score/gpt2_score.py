from heuristic.score import Score
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Tokenizer, GPT2LMHeadModel


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
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.length_normalization = length_normalization
        self.batch_size = batch_size
        self.verbose = verbose

    def compute_score(self, sentences):
        """
        Compute GPT2 score of the sentences
        :param sentences: str | List[str] sentences to evaluate:
        :return: List[float], list of score
        """
        sentences = [sentences] if type(sentences) == str else sentences

        def encode_with_bos_eos_tokens(text):
            # TODO Evaluate if best with + self.tokenizer.eos_token
            return self.tokenizer.encode(self.tokenizer.bos_token + text)

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
            pred_logits = []
            while i + self.batch_size < input_ids.shape[0]:
                if self.verbose:
                    print("\rComputation {:.2f}%".format(i / input_ids.shape[0] * 100), end="")
                pred_logits.append(self.model(input_ids[i: i + self.batch_size])[0])
                i += self.batch_size
            if i < input_ids.shape[0]:
                pred_logits.append(self.model(input_ids[i:])[0])
            if self.verbose:
                print("\rComputation 100%")

        # pred_scores.shape = [len(sentences), max(len(tokenize(sentence)) + 2), vocab_size]
        pred_logits = torch.cat(pred_logits, dim=0)[:, :-1, :]  # To skip prediction for last tokens

        sentences_score = []
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        for i in range(len(sentences)):
            sentence_token_ids = tokens_ids[i][1:]  # len = nb token in sentence + 1 (for BOS)
            sentence_pred_logits = pred_logits[i, :, :]
            sentence_pred_scores = logsoftmax(sentence_pred_logits)
            sentence_token_scores = sentence_pred_scores[range(len(sentence_token_ids)), sentence_token_ids]

            if self.length_normalization:
                sentences_score.append(torch.exp(torch.mean(sentence_token_scores)).item())
            else:
                sentences_score.append(torch.exp(torch.sum(sentence_token_scores)).item())

        return sentences_score
