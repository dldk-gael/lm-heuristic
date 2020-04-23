from cfg_generator import generate_all_sentences
from lm_scorer.lm_scorer.models.auto import AutoLMScorer as LMScorer

import torch
from tqdm import tqdm
import os
import pickle


def rank_sentences(sentences):
    """
    :param sentences : list[str]
    :scorer : function list[str] -> list[(str, float)]
    :return list[(str, float)] ordered list of the string with their scores
    """
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    scorer = LMScorer.from_pretrained("gpt2", device=device)

    scored_sentences = [(sentence, scorer.sentence_score(sentence)) for sentence in tqdm(sentences)]
    sorted_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)

    return sorted_sentences


GRAMMAR_FOLDER = "data/cfg/"
RESULT_FOLDER = "results/"
GRAMMAR_NAME = "ex_1"
X_BEST_RESULTS = 100

if __name__ == '__main__':

    if not os.path.exists(RESULT_FOLDER + GRAMMAR_NAME + '.pkl'):
        sentences = generate_all_sentences(GRAMMAR_FOLDER + GRAMMAR_NAME + '.cfg')
        ranked_sentences = rank_sentences(sentences)
        pickle.dump(ranked_sentences, open(RESULT_FOLDER + GRAMMAR_NAME + '.pkl', 'wb'))

    ranked_sentences = pickle.load(open(RESULT_FOLDER + GRAMMAR_NAME + '.pkl', 'rb'))
    for i, (sentence, score) in enumerate(ranked_sentences):
        print("n°%d (Score : %f) : %s" % (i, score, sentence))
        if i == X_BEST_RESULTS:
            break

