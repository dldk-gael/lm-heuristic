from lm_scorer.lm_scorer.models.auto import AutoLMScorer as LMScorer
import torch


if __name__ == '__main__':
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scorer = LMScorer.from_pretrained("gpt2", device=device)

    print(scorer.sentence_score("London emergency service said that 11 people was sent to hospital.", reduce='gmean'))
    print(scorer.sentence_score("emergency service to London has said that 11 people are sent to hospital.", reduce='gmean'))
    print(scorer.sentence_score("The London emergency services said that 11 people had been sent to hospital.", reduce='gmean'))
