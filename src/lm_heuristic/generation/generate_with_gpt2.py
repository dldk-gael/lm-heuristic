from typing import List
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


class GenerateWithGPT2:
    """
    Class to quickly generate sentences from GPT2 model
    Will be remove later
    """

    def __init__(self, model_name: str = "gpt2", max_length: int = 30):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        self.model = GPT2LMHeadModel.from_pretrained(model_name, pad_token_id=self.tokenizer.eos_token_id)
        self.model.eval()
        self.model.to(self.device)
        self.max_length = max_length

    def __call__(self, context_input: str, nb_samples: int = 1) -> List[str]:
        input_ids = self.tokenizer.encode(context_input, return_tensors="pt").to(self.device)
        input_size = input_ids.shape[1]
        outputs_ids = self.model.generate(
            input_ids=input_ids,
            num_return_sequences=nb_samples,
            do_sample=True,
            top_p=0.9,
            max_length=self.max_length + input_size,
        )
        outputs_str = []
        for i in range(nb_samples):
            output = self.tokenizer.decode(outputs_ids[i, input_size:], skip_special_tokens=True)
            outputs_str.append(output)
        return outputs_str

