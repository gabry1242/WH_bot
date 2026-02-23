from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize a BERT model for binary classification
model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

print(model.config)

# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# sample_text = "I absolutely loved this movie! Highly recommend it."
# tokens = tokenizer(sample_text, padding="max_length", truncation=True, max_length=128)

# print(tokens)