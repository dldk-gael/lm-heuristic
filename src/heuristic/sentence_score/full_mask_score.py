from heuristic.sentence_score import Score
import torch


class FullMaskScore(Score):
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
    def __init__(self, model, tokenizer, batch_size=1):
        """
        Initialize the pre-trained BERT model
        :param model: Huggingface pretrained Model
        :param tokenizer: Huggingface Tokenizer
        :param batch_size: int, batch size to use for BERT input
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def compute_score(self, context, sentence):
        """
        Compute BERT score of a sentence
        :param context: str
        :param sentence: str, sentence to evaluate
        :return: float, score
        """
        # In the method, we simply concatenate context and sentence
        sentence = context + sentence

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

        return torch.mean(log_likelihood_scores).item()




