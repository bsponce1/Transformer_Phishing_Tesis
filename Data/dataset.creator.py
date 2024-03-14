# !pip install transformers
import re
import torch
import string
import numpy as np
import pandas as pd
import transformers
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, TensorDataset

import nltk
nltk.download("stopwords")
nltk.download("punkt")

# Define the clean_sentence function to preprocess the text
def clean_sentence(s):
    """Given a sentence, remove its punctuation and stop words."""
    stop_words = set(stopwords.words('english'))
    s = s.translate(str.maketrans('', '', string.punctuation))  # remove punctuation
    tokens = word_tokenize(s)
    cleaned_s = [w for w in tokens if not w in stop_words]  # remove stop-words
    return " ".join(cleaned_s[:30])  # use the first 30 tokens only

# Load the dataset
df = pd.read_csv("Preprocessed_dataset.csv", encoding='latin-1')

# Clean the sentences
df["cleaned_message"] = df["contexts"].apply(clean_sentence)

# Loading pretrained tokenizer and model
tokenizer = transformers.DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = transformers.DistilBertModel.from_pretrained("distilbert-base-uncased")

# Tokenize the sentences adding the special "[CLS]" and "[SEP]" tokens
tokenized = df["cleaned_message"].apply(lambda x: tokenizer.encode(x, add_special_tokens=True, max_length=128, truncation=True))

# Padding and creating attention masks
max_len = max(tokenized.apply(len))
padded = np.array([i + [0]*(max_len-len(i)) for i in tokenized.values])
attention_mask = np.where(padded != 0, 1, 0)

# Convert to PyTorch tensors
input_ids = torch.tensor(padded)
attention_mask = torch.tensor(attention_mask)

# Create a TensorDataset and DataLoader for batch processing
dataset = TensorDataset(input_ids, attention_mask)
batch_size = 16  # Adjust based on your system's memory
dataloader = DataLoader(dataset, batch_size=batch_size)

# Process the input data in batches and collect the encoder hidden states
encoder_hidden_states = []

for batch in dataloader:
    batch_input_ids, batch_attention_mask = batch
    with torch.no_grad():
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask)
        encoder_hidden_states.append(outputs[0][:,0,:].cpu().numpy())

# Concatenate the outputs from all batches
encoder_hidden_states = np.concatenate(encoder_hidden_states, axis=0)

# Create a new dataframe with the encoded features
df_encoded = pd.DataFrame(encoder_hidden_states)

# Insert the original columns at the beginning of the encoded dataframe
df_encoded.insert(loc=0, column='contexts', value=df["contexts"])
df_encoded.insert(loc=0, column='phishing', value=df["phishing"])

# Save the encoded dataset
df_encoded.to_csv("./dataset_encoded.csv", index=False)
