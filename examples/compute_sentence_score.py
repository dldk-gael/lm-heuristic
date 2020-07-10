from lm_heuristic.sentence_score import GPT2Score, BertScore

if __name__ == "__main__":

    # ////////////////////////////////////////////////////////////
    # Demonstration of GPT2-based sentence scorer
    # ////////////////////////////////////////////////////////////
    print("Initialize sentence scorer parameters")
    gpt2_score = GPT2Score(model_name="gpt2", batch_size=2)

    print("Loading the GPT2 model in memory")
    gpt2_score.build()

    print("Computing sentences' score")
    sentences = ["I likes it.", "I like it."]

    gpt2_score.print_sentences_score(sentences)

    context = "Who knows Bas ?"
    gpt2_score.set_context(context)
    sentences = ["I know him.", "I play tennis."]
    print("Computing sentences' score with context %s" % context)
    gpt2_score.print_sentences_score(sentences)

    # ////////////////////////////////////////////////////////////
    # Demonstration of BERT-based sentence scorer
    # ////////////////////////////////////////////////////////////
    print("Initialize sentence scorer parameters")
    bert_score = BertScore(model_name="bert-base-uncased", batch_size=1)

    print("Loading the BERT model in memory")
    bert_score.build()

    print("Computing sentences' score")
    sentences = [
        "It is a private matter between him and me.",
        "It is a private matter between him but me.",
    ]

    bert_score.print_sentences_score(sentences)

