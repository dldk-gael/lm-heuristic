from typing import List, Callable
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm.autonotebook import tqdm
import torch
import tensorflow_hub as hub
import numpy as np


class GPT2Paraphrases:
    def __init__(
        self,
        gpt2_model_name: str = "gpt2-medium",
        sentence_encoder: Callable[[List[str]], List[np.ndarray]] = None,
        paraphasing_context: str = "",
        batch_size: int = 1,
        paraphrase_start_token: str = " = ",
        question_paraphrasing: bool = False,
    ):
        """
        :param gpt2_model_name: distilgpt2, gpt2, gpt2-medium (by-default), gpt2-large or gpt2-xl
        :param sentence_embed: function that embed sentences, 
                will be use to compare meaning between input_sentence and paraphrases
                if not provide will use Universal Sentence Encoder from google
        :param paraphasing_context: context to "indicate to" gpt2 the paraphrasing task
        :param paraphrase_start_token: special token used in paraphrasing context to indicate paraphrasing start
        :param batch_size: use for gpt2 input
        :question_paraphrasing : True if you want to paraphrase question
        """

        # Load in memory the GPT2 model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)

        # Add a pad token to avoid warning during generation
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(
            gpt2_model_name, pad_token_id=self.gpt2_tokenizer.eos_token_id
        )
        self.gpt2_model.eval()
        self.gpt2_model.to(self.device)

        # Load in memory the sentence encoder
        if sentence_encoder is not None:
            self.embed = sentence_encoder
        else:
            self.embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

        self.paraphasing_context_ids = self.gpt2_tokenizer.encode(paraphasing_context)
        self.context_size = len(self.paraphasing_context_ids)
        self.paraphrase_start_token_id = self.gpt2_tokenizer.encode(paraphrase_start_token)
        self.batch_size = batch_size
        self.question_paraphrasing = question_paraphrasing

    def __call__(
        self,
        sentence: str,
        forbidden_words: List[str] = None,
        margin_size: int = 10,
        nb_samples: int = 1,
        top_n_to_keep: int = 1,
    ):
        """
        :param sentence: sentence that need to be paraphrase
        :param forbidden_words: list of words that can not be use for paraphrasing
        :param margin_size: the size of the parapraphrase will be of nb of sentence tokens +/- margin_size
        :param nb_samples: nb of paraphrases to generate
        :param top_n_to_keep: will only keep the top_n_to_keep best paraphrases
        """
        paraphrases = self.generate(sentence, forbidden_words, margin_size, nb_samples)
        paraphrases = self.clean_generations(sentence, paraphrases)
        paraphrases = self.most_close_in_meaning(sentence, paraphrases, top_n_to_keep)

        return paraphrases

    def generate(self, sentence: str, forbidden_words: List[str] = None, margin_size: int = 10, nb_samples: int = 1):
        """
        Split the generation of nb_samples in batch of self.batch_size
        """
        output = []
        for i in tqdm(range(0, nb_samples, self.batch_size)):
            if i + self.batch_size <= nb_samples:
                batch_size = self.batch_size
            else:
                batch_size = nb_samples % self.batch_size
            output += self.generate_single_batch(sentence, forbidden_words, margin_size, batch_size)

        return output

    def generate_single_batch(
        self, sentence: str, forbidden_words: List[str] = None, margin_size: int = 10, batch_size: int = 1
    ):
        sentence_ids = self.gpt2_tokenizer.encode(sentence)

        # Compute the min/max size allowed for generation
        sentence_size = len(sentence_ids)
        min_length = self.context_size + sentence_size + 1 + sentence_size - margin_size
        max_length = self.context_size + sentence_size + 1 + sentence_size + margin_size

        # Encode the full context
        input_ids = torch.tensor(  # pylint: disable=not-callable
            [self.paraphasing_context_ids + sentence_ids + self.paraphrase_start_token_id], device=self.device
        )

        # Encode the forbidden words
        if forbidden_words:
            forbidden_words_ids = [self.gpt2_tokenizer.encode(word, add_prefix_space=True) for word in forbidden_words]
        else:
            forbidden_words_ids = None

        # Generate
        with torch.no_grad():
            outputs_ids = self.gpt2_model.generate(
                input_ids=input_ids,
                num_return_sequences=batch_size,
                do_sample=True,
                top_p=0.9,
                min_length=min_length,
                max_length=max_length,
                bad_words_ids=forbidden_words_ids,
            )

        outputs_str = []

        # We only retrieve the part corresponding to the generation
        for i in range(batch_size):
            output = self.gpt2_tokenizer.decode(outputs_ids[i, input_ids.shape[1] :], skip_special_tokens=True)
            outputs_str.append(output)

        return outputs_str

    def clean_generations(self, input_sentence: str, paraphrases: List[str]):
        """
        1/ Only keep the first sentence of each generation.
        2/ Remove duplicates
        3/ Remove the input sentence if present in the generated paraphrases
        """
        punctuation = "?" if self.question_paraphrasing else "."
        clean_paraphrases = [paraphrase.split(punctuation)[0].strip() + punctuation for paraphrase in paraphrases]
        clean_paraphrases = list(set(clean_paraphrases))
        if input_sentence in clean_paraphrases:
            clean_paraphrases.remove(input_sentence)
        return clean_paraphrases

    def most_close_in_meaning(self, input_sentence: str, paraphrases: List[str], top_n: int = 1):
        """
        Return the top_n paraphrases that are the more close to the input_sentence (in a meaning way)
        """
        enc_input_sentence, *enc_paraphrases = self.embed([input_sentence] + paraphrases)
        scored_paraphrases = [
            (paraphrase, np.inner(enc_input_sentence, enc_paraphrase))
            for (paraphrase, enc_paraphrase) in zip(paraphrases, enc_paraphrases)
        ]
        top_n_paraphrases = sorted(scored_paraphrases, key=lambda x: x[1], reverse=True)[:top_n]

        return [x[0] for x in top_n_paraphrases]

    def successive_forbidden_stratey(
        self, sentence: str, margin_size: int = 10, nb_samples_per_word: int = 1, top_n_to_keep_per_word: int = 1,
    ):
        """
        1/ tokenize the sentences using space.
        2/ for each sentence words, 
               - forbid to use this particular words
               - generate nb_samples_per_word paraphrases
               - retains top_n_to_keep_per_word paraphrases 
        3/ return the concatenation of all paraphrases

        :param sentence: sentence that need to be paraphrase
        :param forbidden_words: list of words that can not be use for paraphrasing
        :param margin_size: the size of the parapraphrase will be of nb of sentence tokens +/- margin_size
        :param nb_samples_per_word: nb of paraphrases to generate for each call 
        :param top_n_to_keep_per_word: will only keep the top_n_to_keep for each call 
        """
        paraphrases = []
        for word in sentence.split(" "):
            paraphrases += self.__call__(sentence, [word], margin_size, nb_samples_per_word, top_n_to_keep_per_word)
        return paraphrases
