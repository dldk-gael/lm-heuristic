from .sentence_score import SentenceScore
from .bert_score import BertScore
from .gpt2_score import GPT2Score

class NulScorer(SentenceScore):
    def __init__(self):
        SentenceScore.__init__(self, "null_scorer")

    def _compute_sentences_scores(self, sentences):
        return len(sentences) * [0.0]
    
    def build(self):
        return 

test_scorer = NulScorer()