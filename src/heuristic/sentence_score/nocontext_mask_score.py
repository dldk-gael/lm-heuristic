from heuristic.sentence_score import Score
import torch


class NoContextMaskScore(Score):
    """
    Same idea than in BERT Score. However, the context is know consider apart.
    For instance, if the context is "Where is Gael ?" and a possible sentence is "He has left"
    We will create the following mask sentences
        - [CLS] Where is Gael ? [SEP] [MASK] has left
        - [CLS] Where is Gael ? [SEP] he [MASK] left
        - [CLS] Where is Gael ? [SEP] he has [MASK]

    and use the following tokens_type to match with BERT training mode
          [CLS] Where is Gael ? [SEP] [MASK] has left
    type    1    1     1  1   1   1     0     0   0

    then the sentence score will be the average likelihood of true tokens at the mask position id
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

    def __str__(self):
        return 'NoContextMaskScore_with_' + self.model_name

    def compute_score(self, context, sentence):
        """
        Compute BERT score of context:sentence (for now we only mask token of sentence)
        :param context: str
        :param sentence: str, sentence to evaluate
        :return: float, score
        """
        tok_sentence = self.tokenizer.tokenize(sentence)
        encoded_sentence = self.tokenizer.convert_tokens_to_ids(tok_sentence)

        tok_context = self.tokenizer.tokenize(context)
        mask_sentences = [tok_sentence.copy() for _ in range(len(tok_sentence))]
        for i in range(len(tok_sentence)):
            mask_sentences[i][i] = '[MASK]'

        input_sentences = [['[CLS]'] + tok_context + ['[SEP]'] + mask_sentence + ['[SEP]']
                           for mask_sentence in mask_sentences]

        input_ids = torch.stack([torch.tensor(self.tokenizer.convert_tokens_to_ids(input_sentence))
                                 for input_sentence in input_sentences],
                                dim=0)

        segment_ids = [0] * len(['[CLS]'] + tok_context + ['[SEP]']) + \
                      [1] * len(tok_sentence + ['[SEP]'])

        segments_ids = torch.tensor([segment_ids.copy() for _ in range(len(tok_sentence))])

        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()
            input_ids = input_ids.cuda()
            segments_ids = segments_ids.cuda()

        # Compute all prediction logits by batch
        i = 0
        pred_logits = []
        while i + self.batch_size < input_ids.shape[0]:
            outputs = self.model(input_ids[i: i+self.batch_size], token_type_ids=segments_ids)
            pred_logits.append(outputs[0].detach().cpu())
            i += self.batch_size

        if i < input_ids.shape[0]:
            outputs = self.model(input_ids[i:], token_type_ids=segments_ids)
            pred_logits.append(outputs[0].detach().cpu())

        all_pred_logits = torch.cat(pred_logits, dim=0)

        # retrieve only logits corresponding to mask tokens
        begin_sentence_idx = 1 + len(tok_context) + 1  # nb tokens in context + CLS + SEP
        mask_positions = range(begin_sentence_idx, begin_sentence_idx + len(tok_sentence))
        mask_pred_logits = all_pred_logits[range(input_ids.shape[0]), mask_positions, :]

        # for debug mode, check which logit are predicted each time
        mask_pred_token_id = torch.argmax(mask_pred_logits, dim=1)
        mask_pred_tokens = self.tokenizer.convert_ids_to_tokens(mask_pred_token_id.tolist())

        # compute log_likelihood and retrieve value for true_tokens
        log_likelihood_scores = torch.nn.LogSoftmax(dim=1)(mask_pred_logits)
        log_likelihood_scores = log_likelihood_scores[range(input_ids.shape[0]), encoded_sentence]

        return torch.mean(log_likelihood_scores).item()






