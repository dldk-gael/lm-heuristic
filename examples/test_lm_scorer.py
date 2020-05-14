from lm_scorer.models.gpt2 import GPT2LMScorer

if __name__ == "__main__":
    scorer_batch_1 = GPT2LMScorer("gpt2", batch_size=1)
    sentences = "I am good"
    
    scorer_batch_1.sentence_score(sentences, reduce='gmean')

    scorer_batch_4 = GPT2LMScorer("gpt2", batch_size=4)

    print(scorer_batch_4.sentence_score(sentences, reduce='gmean'))



