from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_model_tokenizer(model_id: str = "roberta-base"):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.eval().to("cuda")
    return model, tok
