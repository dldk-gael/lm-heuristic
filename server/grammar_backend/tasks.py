from grammar_backend import celery
from lm_heuristic.generation import GPT2Paraphrases

PATH_TO_PARAPHRASE_CONTEXT = "../data/text/paraphrase.txt"
GPT2_MODEL_TO_USE = "gpt2"
BATCH_SIZE = 1
paraphrase_generator = None


def initialize_paraphraser():
    global paraphrase_generator
    with open(PATH_TO_PARAPHRASE_CONTEXT, "r") as file:
        paraphrase_context = file.read()
    paraphrase_generator = GPT2Paraphrases(
        GPT2_MODEL_TO_USE, paraphasing_context=paraphrase_context, question_paraphrasing=False, batch_size=BATCH_SIZE,
    )

@celery.task(bind=True)
def compute_paraphrase(self, data):
    if not paraphrase_generator:
        initialize_paraphraser()
        
    self.update_state(state="PROGRESS")
    paraphrases = paraphrase_generator(
        sentence=data["sentence_to_paraphrase"],
        forbidden_words=data["forbidden_words"],
        nb_samples=data["number_of_samples"],
        top_n_to_keep=data["keep_top"],
    )
    return paraphrases
