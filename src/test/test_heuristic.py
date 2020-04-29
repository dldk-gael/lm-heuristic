from heuristic import GPT2Score, BertScore
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == "__main__":
    print("Loading the transformers model in memory")
    gpt2_score = GPT2Score(model_name='gpt2', batch_size=2, length_normalization=True)

    print("Computing sentences's score")

    print(gpt2_score.compute_score("London emergency service said that 11 people was sent to hospital."))
    print(gpt2_score.compute_score("emergency service to London has said that 11 people are sent to hospital."))
    print(gpt2_score.compute_score("The London emergency services said that 11 people had been sent to hospital."))
