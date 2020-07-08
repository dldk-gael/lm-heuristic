import os
import pickle


def load_unigram(model_name):
    abs_path = os.path.dirname(os.path.abspath(__file__))

    if "gpt" in model_name:
        pkl_name = "gpt-openwebtext.pickle"
    elif "bert" in model_name:
        if "uncased" in model_name:
            pkl_name = "bert-uncased-bookcorpus-wikipedia.pickle"
        else:
            pkl_name = "bert-cased-bookcorpus-wikipedia.pickle"
    else:
        raise NotImplementedError("Sentence scorer only work with gpt2-based and BERT-based model")

    return pickle.load(open(os.path.join(abs_path, pkl_name), "rb"))
