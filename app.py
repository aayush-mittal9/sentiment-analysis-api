import uvicorn
from fastapi import FastAPI
import joblib
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from pydantic import BaseModel

import pandas as pd
import re
import nltk
import numpy as np

import string
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('airline_sentiment_analysis.csv')
df = df.loc[:,df.columns!="Unnamed: 0"]
df['airline_sentiment'] = df['airline_sentiment'].map({'positive': 1, 'negative': 0})

from sklearn.base import BaseEstimator, TransformerMixin

class CleanText(BaseEstimator, TransformerMixin):
    def remove_mentions(self, input_text):
        return re.sub(r'@\w+', '', input_text)
    
    def remove_urls(self, input_text):
        return re.sub(r'http.?://[^\s]+[\s]?', '', input_text)
    
    def remove_punctuation(self, input_text):
        # Make translation table
        punct = string.punctuation
        trantab = str.maketrans(punct, len(punct)*' ')  # Every punctuation symbol will be replaced by a space
        return input_text.translate(trantab)

    def remove_digits(self, input_text):
        return re.sub('\d+', '', input_text)
    
    def to_lower(self, input_text):
        return input_text.lower()
    
    def remove_stopwords(self, input_text):
        stopwords_list = stopwords.words('english')
        # Some words which might indicate a certain sentiment are kept via a whitelist
        whitelist = ["n't", "not", "no"]
        words = input_text.split() 
        clean_words = [word for word in words if (word not in stopwords_list or word in whitelist) and len(word) > 1] 
        return " ".join(clean_words) 
    
    
    def fit(self, X, y=None, **fit_params):
        return self
    
    def transform(self, X, **transform_params):
        clean_X = X.apply(self.remove_mentions).apply(self.remove_urls).apply(self.remove_punctuation).apply(self.remove_digits).apply(self.to_lower).apply(self.remove_stopwords)
        return clean_X


ct = CleanText()
sr_clean = ct.transform(df.text)

df.text=sr_clean

text = df['text']

texts = []
for i in range(len(text)):
  texts.append(text[i])

MAX_NB_WORDS = 40000
MAX_SEQUENCE_LENGTH = 21

tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# models
# sentiment_model = open("best_model2.hdf5", "rb")
# sentiment_clf = joblib.load(sentiment_model)

# store = HDFStore('best_model2.hdf5')

# model.load_weights("weights.best.hdf5")

from keras.models import load_model
sentiment_clf = load_model('best_model2.hdf5')

class ARCP(BaseModel):
    text1: str

# init app
app = FastAPI()

# Routes
@app.get('/')
async def index():
    return {"text": "Sentiment Analysis `By Aayush Mittal"}

@app.post('/predict/')
async def predict(text: ARCP):
    max_len=21
    sentiment = ['Negative','Positive']

    sequence = tokenizer.texts_to_sequences([text.text1])
    test = pad_sequences(sequence, maxlen=max_len)
    a = sentiment[np.around(sentiment_clf.predict(test), decimals=0).argmax(axis=1)[0]]

    return{ "prediction":a}

if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)