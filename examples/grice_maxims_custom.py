import pandas as pd
from lm_heuristic.sentence_score import GPT2Score

grice_dataset = pd.read_csv("data/grice_dataset.csv", sep=";")

gpt2_scorer = GPT2Score("gpt2", batch_size=2, length_normalization=True)
gpt2_scorer.build()
sentence = "\" Who is gael ? \" \n\" Gael is a friend. \""
print(sentence)
ids = gpt2_scorer.tokenizer.encode(sentence)
print(ids)
print(gpt2_scorer.tokenizer.convert_ids_to_tokens(ids))