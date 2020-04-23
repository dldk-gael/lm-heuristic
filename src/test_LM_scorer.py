from lm_scorer.lm_scorer.models.auto import AutoLMScorer as LMScorer
import torch


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scorer = LMScorer.from_pretrained("gpt2", device=device)

    print(scorer.sentence_score("this is a test", reduce='gmean'))

