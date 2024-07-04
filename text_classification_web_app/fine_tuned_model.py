from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import DistilBertTokenizerFast
from datasets import Dataset, DatasetDict
import os
import pandas as pd

# Set the Hugging Face token as an environment variable
HUGGINGFACE_TOKEN = "enter_your_huggingface_token_here"
os.environ["HUGGINGFACE_TOKEN"] = HUGGINGFACE_TOKEN

# Load the tokenizer and model
model_name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)
model = DistilBertForSequenceClassification.from_pretrained(model_name, use_auth_token=HUGGINGFACE_TOKEN)

# Load and prepare the dataset
dataset = pd.read_csv("SPAM_text_message.csv")

# Rename columns to match Hugging Face's expected format
dataset.rename(columns={'Message': 'text', 'Category': 'label'}, inplace=True)

# Ensure labels are integers
label_mapping = {"spam": 1, "ham": 0}
dataset['label'] = dataset['label'].map(label_mapping)

# Convert pandas DataFrame to Hugging Face Dataset
hf_dataset = Dataset.from_pandas(dataset)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_datasets = hf_dataset.map(tokenize_function, batched=True)

# Split the dataset into train and test sets
train_test_split = tokenized_datasets.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]

# Fine-tune the model
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

# Save the model and tokenizer
model.save_pretrained("fine-tuned-model")
tokenizer.save_pretrained("fine-tuned-model")
