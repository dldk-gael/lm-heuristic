from heuristic import GPT2Score, BertScore


if __name__ == "__main__":
    print("Loading the transformers model in memory")
    gpt2_score = GPT2Score(model_name='gpt2', batch_size=2, length_normalization=True)

    print("Computing sentences's score")
    sentences = ["This is a simple test", "This are simple test", "Final sentences"]

    gpt2_score.print_sentences_score(sentences)

    del gpt2_score


    print("Loading the transformers model in memory")
    bert_score = BertScore(model_name='bert-base-uncased', batch_size=1, length_normalization=True)

    print("Computing sentences's score")
    sentences = ["This is a simple test", "This is another dog test"]

    bert_score.print_sentences_score(sentences)

    del bert_score

