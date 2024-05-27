from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, logging


def load_model(model_name):

    logging.set_verbosity_info()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

    return model, tokenizer
