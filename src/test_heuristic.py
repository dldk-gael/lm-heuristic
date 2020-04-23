from heuristic import GPT2Score


if __name__ == "__main__":
    print("Loading the transformers model in memory")
    gpt2_score = GPT2Score(model_name='gpt2', batch_size=1, length_normalization=True)

    print("Computing sentences's score")
    sentences = ["This is a simple test", "This is another dog test"]

    gpt2_score.print_sentences_score(sentences)