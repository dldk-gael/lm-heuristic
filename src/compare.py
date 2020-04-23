from heuristic.utils import compute_score, GPT2_BASE
from lm_scorer.lm_scorer.models.auto import AutoLMScorer as LMScorer
import torch

if __name__ == '__main__':
    sentence = "This is a simple test"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scorer = LMScorer.from_pretrained("gpt2", device=device)

    print(scorer.sentence_score(sentence, reduce='gmean'))

    del scorer

    score = compute_score([{'score_type': 'PerplexityScore', **GPT2_BASE}], sentences=[sentence], context="")