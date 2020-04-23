from abc import ABC


class Score(ABC):
    """
    Abstract class from which all custom sentence score must inheritate
    """
    def __init__(self, **kwargs):
        pass

    def compute_score(self, context, sentence):
        """
        Compute the naturalness score of a sentence given a context,
        bigger the score, better the sentence
        :param context: str
        :param sentence: str
        :return: float, score
        """
        pass

    def rank_sentences(self, context, sentences):
        """
        Rank the sentences by their BERT scores
        :param context: str
        :param sentences: list[str]
        :return: list[(str, float)] sorted list of sentence with their associated BERT scores
        """
        scored_sentences = [(sentence, self.compute_score(context, sentence)) for sentence in sentences]
        return sorted(scored_sentences, key=lambda x: x[1], reverse=True)
