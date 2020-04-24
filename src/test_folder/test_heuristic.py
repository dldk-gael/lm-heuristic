from heuristic import GPT2Score, BertScore
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    print("Loading the transformers model in memory")
    gpt2_score = GPT2Score(model_name='gpt2', batch_size=2, length_normalization=True)

    print("Computing sentences's score")
    sentences = ["I like it.", "I like it"]

    gpt2_score.print_sentences_score(sentences)

    del gpt2_score


    print("Loading the transformers model in memory")
    bert_score = BertScore(model_name='bert-base-uncased', batch_size=1, length_normalization=True)

    print("Computing sentences's score")
    sentences = ["This is a simple test", "This is another dog test"]

    bert_score.print_sentences_score(sentences)

    del bert_score

