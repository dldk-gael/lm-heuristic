from heuristic import sentence_score
import transformers

GPT2_BASE = {'model_type': 'GPT2LMHeadModel', 'tokenizer_type': 'GPT2Tokenizer', 'model_name': 'gpt2'}
GPT2_MEDIUM = {'model_type': 'GPT2LMHeadModel', 'tokenizer_type': 'GPT2Tokenizer', 'model_name': 'gpt2-medium'}

DIALO_GPT_BASE = {'model_type': 'GPT2LMHeadModel', 'tokenizer_type': 'GPT2Tokenizer',
                  'model_name': 'microsoft/DialoGPT-small'}
DIALO_GPT_MEDIUM = {'model_type': 'GPT2LMHeadModel', 'tokenizer_type': 'GPT2Tokenizer',
                    'model_name': 'microsoft/DialoGPT-medium'}

BERT_BASE = {'model_type': 'BertForMaskedLM', 'tokenizer_type': 'BertTokenizer', 'model_name': 'bert-base-uncased'}


# noinspection PyUnresolvedReferences
def compute_score(scores, context, sentences):
    """
    Will instanciate the score_LM object and run the evaluation for each score_LM / sentences
    :param scores: list of dict {'score_type':str, 'model_type': str, 'tokenizer_type': str, 'model_name': str}
    :param context: str
    :param sentences: str
    """
    print("Context :\n", context)
    print("Possibles sentences:")
    for sentence in sentences:
        print(sentence)
    print("-------")

    for score_dict in scores:
        print("Loading %s in memory" % score_dict['model_name'])
        model = transformers.__getattribute__(score_dict['model_type']).from_pretrained(score_dict['model_name'])
        tokenizer = transformers.__getattribute__(score_dict['tokenizer_type']).from_pretrained(score_dict['model_name'])
        score = sentence_score.__getattribute__(score_dict['score_type'])(model, tokenizer)
        print("Ranking sentences using %s with %s model" % (score_dict['score_type'], score_dict['model_name']))

        for i, (sentence, score_result) in enumerate(score.rank_sentences(context, sentences)):
            print("\tn°%d (score : %f) - %s" % (i + 1, score_result, sentence))
        print("")

        del model, tokenizer, score  # to save memory for laptop ...
