
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import evaluate
import pandas as pd

file_path = "IMDB Dataset.csv"

file = pd.read_csv(file_path)

file_train = file.iloc[:5000]
file_test = file.iloc[5000:5500]

raw_data = {
    "text": file_train["review"].tolist(),
    "label": file_train["sentiment"].tolist()
}

test_data = {
    "text": file_test["review"].tolist(),
    "label": file_test["sentiment"].tolist()
}

# 1. Load tokenizer and model
model_name = "google-bert/bert-base-uncased"
print(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
print(model)

# for param in model.bert.parameters():
#   param.requires_grad = False

# 2. Prepare sample dataset

dataset = Dataset.from_dict(raw_data)
test_dataset = Dataset.from_dict(test_data)

# 3. Tokenize function
def tokenize_function(examples):
  return tokenizer(examples["text"], padding="max_length", truncation=True)

def compute_metrics(eval_pred):
  metric_acc = evaluate.load("accuracy")
  metric_f1 = evaluate.load("f1")

  logits, labels = eval_pred
  predictions = np.argmax(logits, axis=-1)

  acc = metric_acc.compute(predictions=predictions, references=labels)
  f1 = metric_f1.compute(predictions=predictions, references=labels)

  return {**acc, **f1}

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_eval_datasets = test_dataset.map(tokenize_function, batched=True)

# 4. Set training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="no",  # use "epoch" if validation data is provided
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
)

# 1. Count the parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# 2. Print the results
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Frozen Parameters: {total_params - trainable_params:,}")


# 5. Initialize Trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    eval_dataset = tokenized_eval_datasets,
    compute_metrics = compute_metrics
)

trainer.train()
metrics = trainer.evaluate()
print(metrics)

trainer.model.save_pretrained('./saved_model_more_data/')
tokenizer.save_pretrained('./saved_model_more_data/')