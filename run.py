import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# Load trained model (NOT base model)
model_path = "./final_model"   # or wherever Trainer saved it
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

model.eval()

def predict_class(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred = int(np.argmax(probs))
    return {"pred_label": pred, "prob_0": float(probs[0]), "prob_1": float(probs[1])}

print(predict_class("ok sigma"))