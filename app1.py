import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
from collections import Counter
import numpy as np
import math

# Ensure you have the necessary NLTK resources
nltk.download('punkt')


import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import wordnet
from fastai.text.all import *

import streamlit as st
import torch
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from transformers import T5Tokenizer,T5ForConditionalGeneration
model_name = "t5-base"  # You can also try "t5-large" or "t5-xxl" if available
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name).to('cuda' if torch.cuda.is_available() else 'cpu')

# Load the saved model and encoder
learn = load_learner('export.pkl')
# learn = load_learner('/kaggle/working/export.pkl')
learn.load_encoder('ft_enc')
def stylometric_features(text):
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    unique_words = set(words)
    
    avg_sentence_length = sum(len(sentence.split()) for sentence in sentences) / len(sentences)
    lexical_density = len(unique_words) / len(words)
    avg_word_length = sum(len(word) for word in words) / len(words)
    
    return {
        "avg_sentence_length": avg_sentence_length,
        "lexical_density": lexical_density,
        "avg_word_length": avg_word_length,
    }
def calculate_perplexity(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors='pt').to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Get the model's output
    with torch.no_grad():  # Disable gradient calculation to save memory
        outputs = model(**inputs, labels=inputs['input_ids'])
    
    # Calculate perplexity
    loss = outputs.loss
    perplexity = torch.exp(loss).item()
    return perplexity

def calculate_burstiness(text):
    tokens = text.split()  # Simple tokenization by whitespace
    token_counts = {}
    
    # Count occurrences of each token
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1
            
    # Calculate burstiness
    counts = np.array(list(token_counts.values()))
    mean = counts.mean()
    
    # Calculate burstiness as the ratio of the standard deviation to the mean
    if mean == 0:
        return 0  # Avoid division by zero
    std_dev = counts.std()
    
    # Calculate burstiness using a more structured approach
    burstiness = std_dev / mean
    
    # Additionally, we can calculate the burstiness based on the frequency of the most common tokens
    # This captures the idea of "bursty" behavior more effectively
    most_common_count = counts.max()
    burstiness += (most_common_count - mean) / mean  # Add a factor for the most common token
    
    return burstiness
def predict_class(user_input):
    # Create a DataFrame for the user input
    input_df = pd.DataFrame({'text': [user_input]})
    
    # Create a DataLoader from the DataFrame
    input_dl = learn.dls.test_dl(input_df)
    
    # Make predictions
    preds = learn.get_preds(dl=input_dl)
    
    # Get the predicted class index and probability
    pred_idx = preds[0].argmax(dim=1).item()  # Get the index of the highest probability
    pred_prob = preds[0][0][pred_idx].item()
    perplexity = calculate_perplexity(user_input)
    burstiness = calculate_burstiness(user_input)
    features = stylometric_features(user_input)
    print(perplexity,burstiness,features,pred_prob)
    if pred_prob > 0.98:
        return pred_idx
    else:
        if (perplexity < 20 and burstiness > 1.5) and ( features['lexical_density']>0.5 and features['avg_word_length']<6):
            return 1 #LLM Generated
        elif (perplexity > 20 and burstiness > 1.5) and ( features['lexical_density']>0.5 and features['avg_word_length']<6):
            return 2  # LLM Rewritten
        elif (perplexity > 20 and burstiness < 1.5) or ( features['lexical_density']>0.5 and features['avg_word_length']<6):
            return 0  # Human Generated

# Define a function to analyze the percentage of AI-generated tokens in the text
def analyze_ai_generated_percentage(text):
    # Tokenize the input text
    tokens = word_tokenize(text)
    
    # Count the total number of tokens
    total_tokens = len(tokens)
    
    # Count the number of AI-generated tokens
    ai_generated_count = sum(predict_token_class(token) for token in tokens)
    
    # Calculate the percentage of AI-generated tokens
    if total_tokens > 0:
        percentage_ai_generated = (ai_generated_count / total_tokens) * 100
    else:
        percentage_ai_generated = 0.0  # Avoid division by zero
    
    return percentage_ai_generated


# Streamlit app layout
st.title("ಪತ್ತೆ.ai")
st.write("Enter text to analyze the percentage of AI-generated tokens.")

# Text input area
user_input = st.text_area("Input Text", height=300)


# Button to analyze the text
if st.button("Analyze"):
    if user_input:
#         percentage_ai_generated = analyze_ai_generated_percentage(user_input)
        if predict_class(user_input) == 1:
            st.write("**Result:** AI Generated")
        elif predict_class(user_input) == 2:
            st.write("**Result:Mostly LLM Re-Written Text")
        else:
            st.write("**Result:** Human Generated")
        
    else:
        st.write("Please enter some text to analyze.")

# print(predict_class("""
# Numerous studies demonstrate that using a cell phone while driving poses significant dangers and can result in loss of life. Research indicates that engaging in 50 minutes of phone conversation while driving increases the risk of a car crash nearly five-fold (Crundall, 2017). While texting is also a concern, many reported incidents highlight the dangers of talking on a cell phone while behind the wheel. In the United States, 20 states have prohibited the use of hand-held cell phones, and 48 states have banned texting while driving (National Conference of State Legislatures [NCSL], 2019). Surprisingly, no state has implemented laws that broadly prohibit cell phone use among drivers, which contributes to the ongoing increase in accidents caused by cell phone distractions. Many drivers acknowledge that "distractions from mobile phones" are a primary cause of car and road accidents.
# One significant issue is the danger posed by using one hand to navigate turns and respond to potential hazards while conversing on the phone. According to Crundall (2017), a substantial number of road accidents in Europe occur on steep slopes, bends, and corners, with 35% involving drivers who were communicating on their cell phones. Additionally, engaging in conversation can lead to diminished concentration and cognitive processing, ultimately resulting in accidents (NCSL, 2019). A study conducted by scientists at Carnegie Mellon University found that dual-tasking negatively impacts human spatial brain activity, leading to decreased focus and increased risk of accidents (Crundall, 2017). The challenge of processing auditory messages while driving further underscores the necessity for drivers to avoid using their cell phones under any circumstances.
# """))
