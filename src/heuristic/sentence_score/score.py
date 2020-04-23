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
        :return: float | List[float], score
        """
        ...

    def rank_sentences(self, sentences):
        """
        Rank the sentences by their scores
        :param sentences: list[str]
        :return: list[(str, float)] sorted list of sentence with their associated LM scores
        """
        return sorted(self.compute_scores(sentences), key=lambda x: x[1], reverse=True)
