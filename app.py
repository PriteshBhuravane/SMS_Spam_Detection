import streamlit as st
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ✅ Download required NLTK data safely (only if missing)
for resource in ["punkt", "stopwords"]:
    try:
        nltk.data.find(f"tokenizers/{resource}" if resource == "punkt" else f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ✅ Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# ✅ Streamlit UI
st.title('Email/SMS Spam Classifier')

input_text = st.text_area('Enter your message')

if st.button('Predict'):
    # Preprocess
    transformed_sms = transform_text(input_text)

    # Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # Predict
    result = model.predict(vector_input)

    if result == 1:
        st.header('🚨 Spam')
    else:
        st.header('✅ Not Spam')
