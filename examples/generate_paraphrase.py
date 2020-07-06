from lm_heuristic.generation import GPT2Paraphrases

if __name__ == "__main__":
    BATCH_SIZE = 1
    NB_SAMPLES = 10
    TOP_TO_KEEP = 2

    # EXAMPLE WITH AFFIRMATIVES SENTENCES
    PAPAPHRASE_FILE = "data/paraphrase_seed/affirmative_paraphrases.txt"
    INPUT_SENTENCE = "I like to play tennis outside when the weather is good."
    with open(PAPAPHRASE_FILE, "r") as file:
        affirmative_paraphrase_context = file.read()

    paraphrase_generator = GPT2Paraphrases(
        "gpt2", paraphasing_context=affirmative_paraphrase_context, batch_size=BATCH_SIZE
    )
    paraphrases = paraphrase_generator(INPUT_SENTENCE, margin_size=5, nb_samples=NB_SAMPLES, top_n_to_keep=TOP_TO_KEEP)

    print("INPUT SENTENCE :", INPUT_SENTENCE)
    print("PARAPRHASES :")
    for i, paraphrase in enumerate(paraphrases):
        print("n°%d : %s" % (i, paraphrase))

    # EXAMPLE WITH QUESTIONS
    QUESTION_FILE = "data/paraphrase_seed/affirmative_paraphrases.txt"
    INPUT_QUESTION = "What is your name?"

    with open(QUESTION_FILE, "r") as file:
        question_paraphrase_context = file.read()

    paraphrase_generator = GPT2Paraphrases(
        "gpt2",
        paraphasing_context=question_paraphrase_context,
        question_paraphrasing=True,
        batch_size=BATCH_SIZE,
    )

    paraphrases = paraphrase_generator(INPUT_QUESTION, margin_size=5, nb_samples=NB_SAMPLES, top_n_to_keep=TOP_TO_KEEP)

    print("INPUT SENTENCE :", INPUT_QUESTION)
    print("PARAPRHASES :")
    for i, paraphrase in enumerate(paraphrases):
        print("n°%d : %s" % (i, paraphrase))
