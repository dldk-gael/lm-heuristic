from lm_heuristic.generation import GPT2Paraphrases

if __name__ == "__main__":
    PAPAPHRASE_FILE = "data/text/paraphrase.txt"
    INPUT_SENTENCE = "I like to play tennis outside when the weather is good."

    with open(PAPAPHRASE_FILE, "r") as file:
        paraphrase_context = file.read()

    paraphrase_generator = GPT2Paraphrases("gpt2", paraphasing_context=paraphrase_context, batch_size=2)
    paraphrases = paraphrase_generator(INPUT_SENTENCE, margin_size=3, nb_samples=20, top_n_to_keep=5)

    for paraphrase in paraphrases:
        print(paraphrase)
