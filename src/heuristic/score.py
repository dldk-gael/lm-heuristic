from abc import ABC, abstractmethod
from typing import *


class Score(ABC):
    """
    A Score object compute a "naturalness" score for a sentence or a list of sentences
    """

    def __init__(self, **kwargs):
        ...

    @overload
    def compute_score(self, text: str) -> float:
        ...

    @overload
    def compute_score(self, text: List[str]) -> List[float]:
        ...

    @abstractmethod
    def compute_score(
        self, text: Union[str, List[str]]
    ) -> Union[str, List[float]]:
        """
        Compute the naturalness score of sentences
        :param text: single sentence or list of sentences
        :return: sentence's score or  list of sentence's scores
        """
        ...

    def rank_sentences(self, sentences: List[str]) -> List[Tuple[str, float]]:
        """
        Rank the sentences by their scores
        :param sentences: list[str]
        :return: sorted list of sentence with their associated scores
        """
        scored_sentences = list(zip(sentences, self.compute_score(sentences)))
        return sorted(scored_sentences, key=lambda x: x[1], reverse=True)

    def print_sentences_score(self, sentences: List[str]):
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

    def __call__(self, sentences):
        return self.compute_score(sentences)
