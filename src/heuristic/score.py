from abc import ABC, abstractmethod


class Score(ABC):
    """
    Abstract class from which all custom sentence score must inheritate
    """
    def __init__(self, **kwargs):
        ...

    @abstractmethod
    def compute_score(self, sentences):
        """
        Compute the naturalness score of sentence (or list of sentences)
        :param sentences: str | List[str]
        :return: float | List[float], list of sentence scores
        """
        ...

    def rank_sentences(self, sentences):
        """
        Rank the sentences by their scores
        :param sentences: list[str]
        :return: list[(str, float)] sorted list of sentence with their associated scores
        """
        scored_sentences = list(zip(sentences, self.compute_score(sentences)))
        return sorted(scored_sentences, key=lambda x: x[1], reverse=True)

    def print_sentences_score(self, sentences):
        """
        Print each sentence from the most probable to the less probable along with their scores
        """
        self.print_ranked_sentences(self.rank_sentences(sentences))

    @staticmethod
    def print_ranked_sentences(ranked_sentences, n=-1):
        for i, (sentence, score_result) in enumerate(ranked_sentences):
            print("\tn°%d (score : %f) - %s" % (i + 1, score_result, sentence))
            if i == n:
                break
        print("")



