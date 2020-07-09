import pandas as pd
from lm_scorer.models.auto import AutoLMScorer as LMScorer

grice_dataset = pd.read_csv("data/grice_dataset.csv", sep=";")

gpt2_scorer = LMScorer.from_pretrained("gpt2", reduce="gmean")

idx = 0

sentence_1 = grice_dataset.loc[idx]["Questions"] + " " + grice_dataset.loc[idx]["Puppet 1"]
sentence_2 = grice_dataset.loc[idx]["Questions"] + " " + grice_dataset.loc[idx]["Puppet 2"]
print(sentence_1)
print(sentence_2)
scores = gpt2_scorer.sentence_score([sentence_1, sentence_2])
print(scores)
#print(score_1, score_2)