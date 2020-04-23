from heuristic.sentence_score import Score
import torch
import math


class GPT2Score(Score):
    """
    Compute the perplexity score of a sentence given a langage model :
     - average the loglikelihood that the LM assign to each token of the sentence given the previous tokens
    """
    def __init__(self, model, tokenizer, batch_size):
        """
        Initialize the pre-trained GPT2 model
        :param model: Huggingface pretrained Model
        :param tokenizer: Huggingface Tokenizer
        :param batch_size: int
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def compute_score(self, sentences):
        """
        Compute GPT2 score of the sentences
        :param sentences: str | List[str] sentences to evaluate
        :return: List[float], list of score
        """
        sentence = self.tokenizer.bos_token + context + sentence + self.tokenizer.eos_token
        tokens = self.tokenizer.tokenize(sentence)
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor(tokens_ids, dtype=torch.long)

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            input_ids = input_ids.cuda()

        outputs = self.model(input_ids)

        # pred_scores.shape = [1, seq_len + 2, vocab_size]
        pred_logits = outputs[0].detach().cpu()  # shape [1, seq_len + 2, vocab_size]

        tokens_ids = tokens_ids[1:]  # len seq_len + 1
        pred_logits = pred_logits[:-1, :]  # shape [seq_len + 1, vocab_size]
        log_scores = torch.nn.LogSoftmax(dim=1)(pred_logits)

        ids_score = log_scores[range(len(tokens_ids)), tokens_ids]
        average = torch.mean(ids_score).item()
        return math.exp(average)



