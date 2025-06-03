import streamlit as st
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import pipeline
import nltk
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
nltk.download('punkt')
nltk.download('punkt_tab')

try:
    df = pd.read_csv('https://drive.google.com/file/d/1kOluq8NNshQozI5Ik_L9_okq8bSzHiu2/view?usp=sharing')
    #df = pd.read_csv('twcs.csv')
    df = df.sample(n=10000, random_state=42)
except FileNotFoundError:
    st.error("Error: twcs.csv not found.")
    st.stop()

def clean_text(text):
    text = re.sub(r'http\S+', '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    return text
df['cleaned_text'] = df['text'].apply(clean_text)
df['cleaned_text'] = df['cleaned_text'].fillna('')

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
df['sentiment'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment.polarity)
df['issue_category'] = df['sentiment'].apply(lambda x: 'positive' if x > 0 else 'negative' if x < 0 else 'neutral')
model = LogisticRegression(max_iter=1000)
model.fit(X, df['issue_category'])

generator = pipeline('text-generation', model='gpt2', device=0)
analyzer = SentimentIntensityAnalyzer()

def tailored_response(query):
    print(f"Raw query input (length {len(query)}): '{query}'")
    cleaned_query = clean_text(query)
    vader_scores = analyzer.polarity_scores(query)
    compound = vader_scores['compound']
    print(f"VADER compound score: {compound}")
    category = 'positive' if compound > 0.05 else 'negative' if compound < -0.1 else 'neutral'
    if category == 'neutral':
        negative_indicators = ["not", "hasn’t", "delayed", "lost", "damaged", "disappointed"]
        if any(ind in query.lower() for ind in negative_indicators):
            category = 'negative'
        else:
            category = model.predict(vectorizer.transform([cleaned_query]))[0]
    prompt = f"{query} - Respond with a short, empathetic support message specific to the issue."
    response = generator(
        prompt,
        max_length=50,
        num_return_sequences=1,
        truncation=True,
        temperature=0.5,
        top_k=40,
        no_repeat_ngram_size=2
    )
    generated_text = response[0]['generated_text']
    print(f"Raw generated text: {generated_text}")
    cleaned_response = generated_text.replace(prompt, "").strip()
    negative_keywords = ["package", "lost", "order", "delayed", "product", "broken", "arrived", "disappointed", "damaged"]
    positive_keywords = ["love", "thank", "fast", "great", "amazing"]
    relevant_keywords = negative_keywords if category == "negative" else positive_keywords
    is_relevant = any(kw in cleaned_response.lower() for kw in relevant_keywords)
    if len(cleaned_response.split()) < 5 or not is_relevant or '‡' in cleaned_response:
        if category == "negative":
            issue = "lost package" if "lost" in query.lower() or "arrived" in query.lower() else "delay" if "delayed" in query.lower() else "damaged item" if "damaged" in query.lower() else "broken product" if "broken" in query.lower() else "disappointment" if "disappointed" in query.lower() else "issue"
            cleaned_response = f"We're sorry for the {issue}. Please provide your order number."
        else:
            cleaned_response = "Thank you for your kind words! We're glad you're happy."
    return category, cleaned_response

st.title('Customer Support Automation Chatbot')
st.write("Enter a customer query to get an automated response.")
query = st.text_input('Enter your query (e.g., "My order is delayed"):', key="query_input")
if st.button("Submit"):  # Add button to trigger response
    if query is None or len(query.strip()) == 0:
        st.write("**Predicted Issue Category**: Unknown")
        st.write("**Response**: Please enter a valid query.")
    else:
        category, response = tailored_response(query)
        st.write(f"**Predicted Issue Category**: {category}")
        st.write(f"**Response**: {response}")

st.subheader("Issue Trends")
import matplotlib.pyplot as plt
import seaborn as sns
issue_counts = df['issue_category'].value_counts()
fig, ax = plt.subplots()
sns.barplot(x=issue_counts.index, y=issue_counts.values, ax=ax)
ax.set_title("Customer Issue Distribution")
st.pyplot(fig)
