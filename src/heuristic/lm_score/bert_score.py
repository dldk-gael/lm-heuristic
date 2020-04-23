from heuristic.score import Score
from transformers import BertForMaskedLM, BertTokenizer
import torch
import math


class BertScore(Score):
    """
    Use BERT to score a sentence following the idea describe in the paper
    Effective Sentence Scoring Method Using BERT for Speech Recognition. J. Shin, Y. Lee, Kyomin Jung

    Roughly the idea is to :
    1- mask successively each word of the sentences
        For instance, if the context is "Where is Gael ?" and a possible sentence is "He has left"
        We will create the following mask sentences
            - [CLS] [MASK]  is Gael ?  He has left [SEP]
            - [CLS] Where [MASK]  Gael ? he has left [SEP]
            - ...
            - ...
            - [CLS] Where is Gael ? [SEP] he has [MASK] [SEP]

    2- compute the likelihood of each target word that has been mask using context from both side
    3- return the average of all log-likelihood  (in the paper, the authors said they use the sum)
    """
    def __init__(self, model_name, batch_size=1, length_normalization=False):
        """
        Initialize the pre-trained BERT model
        :param model_name : [str] for instance 'bert-base-uncased'
        :param batch_size: int, batch size to use for BERT input
        :param length_normalization [boolean]
        """
        super().__init__()
        self.model = BertForMaskedLM.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.batch_size = batch_size
        self.length_normalization = length_normalization

    def compute_score_single_sentence(self, sentence):
        """
        Compute BERT score of a sentence
        :param sentence
        :return: float, score
        """
        # prepare the batch of mask sentences
        tok_sentence = self.tokenizer.tokenize(sentence)
        encoded_sentence = self.tokenizer.convert_tokens_to_ids(tok_sentence)
        seq_len = len(tok_sentence)

        mask_sentences = [tok_sentence.copy() for _ in range(seq_len)]
        for i in range(seq_len):
            mask_sentences[i][i] = '[MASK]'

        input_sentences = [['[CLS]'] + mask_sentence + ['[SEP]'] for mask_sentence in mask_sentences]
        input_ids = torch.stack([torch.tensor(self.tokenizer.convert_tokens_to_ids(input_sentence))
                                 for input_sentence in input_sentences],
                                dim=0)

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            input_ids = input_ids.cuda()

        # Compute all prediction logits by batch
        i = 0
        pred_logits = []
        while i + self.batch_size < seq_len:
            pred_logits.append(self.model(input_ids[i: i+self.batch_size])[0].detach().cpu())
            i += self.batch_size

        if i < seq_len:
            pred_logits.append(self.model(input_ids[i:])[0].detach().cpu())

        all_pred_logits = torch.cat(pred_logits, dim=0)  # shape (seq_len, seq_len, vocab_size)

        # retrieve only logits corresponding to mask tokens : new shape (seq_len, vocab_size)
        mask_positions = range(1, 1+seq_len)  # not take into account first and last special tokens
        mask_pred_logits = all_pred_logits[range(input_ids.shape[0]), mask_positions, :]

        # compute log_likelihood and retrieve value for true_tokens
        log_likelihood_scores = torch.nn.LogSoftmax(dim=1)(mask_pred_logits)
        log_likelihood_scores = log_likelihood_scores[range(input_ids.shape[0]), encoded_sentence]

        if self.length_normalization:
            return math.exp(torch.mean(log_likelihood_scores).item())
        else:
            return math.exp(torch.sum(log_likelihood_scores).item())

    def compute_score(self, sentences):
        sentences = [sentences] if type(sentences) == str else sentences
        return [(sentence, self.compute_score_single_sentence(sentence)) for sentence in sentences]

