import streamlit as st
import io
import pickle
import numpy as np
import tensorflow as tf
#from tensorflow.keras.models import load_model
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
#from tensorflow.keras.models.preprocessing.sequence import pad_sequences
from keras.preprocessing.sequence import pad_sequences
#load the trainedmodel
model = load_model("SentimentModel.h5")

#load the saved tokenizer used during the training ,the tokenizer 
#stores the 700 vocabulary
with open("tokenizer.pkl", "rb") as tk:
    tokenizer= pickle.load(tk)
#define the function to preprocess the user text input.the text 
# inside the bracket represnt whatever text the user enters
def preprocess_text(text):
    #tokenize the text, take the text & convert to sequence of integers
   tokens = tokenizer.texts_to_sequences([text])

    #pad the sequences to afixed length
   padded_tokens = pad_sequences(tokens, maxlen = 100)
   return padded_tokens[0]

#create the title of the app
st.title("Sentiment Analysis Webapp")
#create a text input widget for user input
user_input = st.text_area("Enter text for sentiment analysis"," ")

#create a button to trigger the sentiment analysis
if st.button("Predict Sentiment"):
    #preprocess the user input
    processed_input = preprocess_text(user_input)

    #make prediction using the loaded model
    prediction = model.predict(np.array([processed_input]))
    st.write(prediction)
    sentiment = "Negative" if prediction[0][0] > 0.5 else "Positive"

    #Display the sentiment
    st.write(f"### Sentiment: {sentiment}")



    
