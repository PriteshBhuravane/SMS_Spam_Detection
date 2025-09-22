import streamlit as st
import string
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ✅ Ensure NLTK resources are available (punkt, punkt_tab, stopwords)
for resource in ["punkt", "punkt_tab", "stopwords"]:
    try:
        if resource.startswith("punkt"):
            nltk.data.find(f"tokenizers/{resource}")
        else:
            nltk.data.find(f"corpora/{resource}")
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

# ✅ Load trained model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))

# ✅ Streamlit UI
st.title('📩 Email/SMS Spam Classifier')

input_text = st.text_area('Enter your message')

if st.button('Predict'):
    if input_text.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_text)

        # Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # Predict
        result = model.predict(vector_input)

        # Show Result
        if result == 1:
            st.error('🚨 Spam Detected!')
        else:
            st.success('✅ This is NOT spam.')
