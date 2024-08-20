import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

word_idx = imdb.get_word_index()
reverse = {value : key for key, value in word_idx.items()}

model = load_model('simple_rnn.h5')
model.summary()

#helper functions
#1 decoder
def decode(encode):
    return ' '.join([reverse.get(i-3, '?') for i in encode])
#2 preprocess users input
def preprocess(text):
    words = text.lower().split()
    encode = [word_idx.get(word,2)+ 3 for word in words]
    padded = sequence.pad_sequences([encode], maxlen=500)
    return padded




import streamlit as st
st.title('movie review Sentiment Analysis')
st.write('enter a movie review to get sentiment analysis as postive or negetive')

user_input = st.text_area('movie review')

if st.button('get review'):
    preprocessed_input = preprocess(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'positive' if prediction > 0.5 else 'negative'
    st.write('sentiment:',sentiment)
    st.write('prediction score:',prediction[0][0])
else:
    st.write('plz enter review')

# prompt: runt this on stramline
