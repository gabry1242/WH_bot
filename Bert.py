import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -------------------------
# 1) Load CSV
# CSV must have: sentence,label  (label is 0/1)
# -------------------------
df = pd.read_csv("processed_dataset.csv")
df = df.dropna(subset=["string", "label"])
df["label"] = df["label"].astype(int)

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)

train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

# -------------------------
# 2) Tokenize
# -------------------------
model_name = "distilbert-base-uncased"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["string"], truncation=True)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

# Rename label column to labels for Trainer
train_ds = train_ds.rename_column("label", "labels")
val_ds = val_ds.rename_column("label", "labels")

# Remove non-input columns (keep sentence only if you want; Trainer ignores unknown cols but cleaner to remove)
train_ds = train_ds.remove_columns([c for c in train_ds.column_names if c not in ["input_ids", "attention_mask", "labels"]])
val_ds = val_ds.remove_columns([c for c in val_ds.column_names if c not in ["input_ids", "attention_mask", "labels"]])

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -------------------------
# 3) Model
# -------------------------
model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# -------------------------
# 4) Metrics
# -------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}

# -------------------------
# 5) Training args (small data friendly)
# -------------------------
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    num_train_epochs=6,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")