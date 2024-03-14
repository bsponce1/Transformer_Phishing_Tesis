from flask import Flask, request
from numpy import array
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Activation, Dropout, Dense
from tensorflow.keras.layers import Flatten, LSTM, GRU, Bidirectional, GlobalAveragePooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D, SpatialDropout1D, GlobalMaxPool1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import nltk
import pandas as pd
import numpy as np
import re

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')


import tensorflow_hub as tf_hub
# Asegúrate de que el path al modelo sea correcto
import tensorflow as tf

model = tf.keras.models.load_model('C:/Users/Bryan/codigo/Codigo_Transformer/Model/BERT_model.h5')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def areceibe_text():
    webpage_text  = [clean_text(request.get_data(as_text=True))]
    
    tokenizer = Tokenizer(num_words=3000)
    tokenizer.fit_on_texts(webpage_text)
    webpage_text = tokenizer.texts_to_sequences(webpage_text)
    vocab_size = len(tokenizer.word_index) + 1
    webpage_text = pad_sequences(webpage_text, padding='post', maxlen=200)
    
    print(webpage_text)
    
    results = model.predict(webpage_text)
    results= results[0][0]
    results = round(results,3)
    print (results)

    return str(results)
    #if results[0][0] > results[0][1]:
    #    return "Pishing"
    #else:
    #    return "Normal"

    
#Lemmatize Words

def fetch_pos_tag(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN


lemmatizer = WordNetLemmatizer()

#cleaning the data now 

regex = [
    r'<[^>]+>', #HTML tags
    r'@(\w+)', # @-mentions
    r"#(\w+)", # hashtags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'[^0-9a-z #+_\\r\\n\\t]', #BAD SYMBOLS
]

REPLACE_URLS = re.compile(r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+')
REPLACE_HASH = re.compile(r'#(\w+)')
REPLACE_AT = re.compile(r'@(\w+)')
REPLACE_HTML_TAGS = re.compile(r'<[^>]+>')
REPLACE_DIGITS = re.compile(r'\d+')
#REPLACE_BY = re.compile(r"[/(){}\[\]\|,;.:?\-\'\"$]")
REPLACE_BY = re.compile(r"[^a-z0-9\-]")

STOPWORDS = set(stopwords.words('english'))

#tokens_re = re.compile(r'('+'|'.join(regex)+')', re.VERBOSE | re.IGNORECASE)

# sentences = [] #for Word2Vec model

def clean_text(text,**args):
    text = text.lower()
    text = REPLACE_HTML_TAGS.sub(' ', text)
    text = REPLACE_URLS.sub('', text)
    text = REPLACE_HASH.sub('', text)
    text = REPLACE_AT.sub('', text)
    text = REPLACE_DIGITS.sub(' ', text)
    text = REPLACE_BY.sub(' ', text)
    text = " ".join(lemmatizer.lemmatize(word.strip(), fetch_pos_tag(pos_tag([word.strip()])[0][1])) for word in text.split() if word not in STOPWORDS and len(word)>3)
    
    #sentences.append(text.split())
    return text



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

