from lm_scorer.models.auto import AutoLMScorer as LMScorer
import torch

"""
Just to show how to use lm_scorer librairy
"""

if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scorer = LMScorer.from_pretrained("gpt2", device=device)

    sentences = [
        "London emergency service said that 11 people was sent to hospital.",
        "emergency service to London has said that 11 people are sent to hospital.",
        "The London emergency services said that 11 people had been sent to hospital.",
    ]

    for sentence in sentences:
        print(scorer.sentence_score(sentence, reduce="gmean"))
