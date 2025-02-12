import streamlit as st
import nltk
from nltk.corpus import reuters, stopwords
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Force re-download of the Reuters dataset
nltk.download('reuters', force=True)
nltk.download('punkt')
nltk.download('stopwords')


# Load Reuters Corpus and Preprocess
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

# Prepare corpus
corpus_sentences = []
document_embeddings = {}
doc_ids = reuters.fileids()

for doc_id in doc_ids:
    raw_text = reuters.raw(doc_id)
    tokenized_text = preprocess_text(raw_text)
    corpus_sentences.append(tokenized_text)
    document_embeddings[doc_id] = tokenized_text

# Train Word2Vec Model
model = Word2Vec(sentences=corpus_sentences, vector_size=100, window=5, min_count=5, workers=4)

def get_average_embedding(words):
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)

# Compute document embeddings
doc_embeddings = {doc_id: get_average_embedding(words) for doc_id, words in document_embeddings.items()}

# Streamlit UI
st.title("🔎 Information Retrieval System with Word2Vec")

query = st.text_input("Enter your search query:")
num_results = st.slider("Number of documents to retrieve:", 1, 10, 5)

def retrieve_documents(query, top_n=5):
    query_tokens = preprocess_text(query)
    query_embedding = get_average_embedding(query_tokens)

    similarities = {}
    for doc_id, doc_embedding in doc_embeddings.items():
        sim_score = cosine_similarity([query_embedding], [doc_embedding])[0][0]
        similarities[doc_id] = sim_score

    # Sort by similarity
    top_docs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return top_docs

if st.button("Search"):
    if query:
        top_results = retrieve_documents(query, num_results)
        st.subheader("📄 Top Relevant Documents:")
        for doc_id, score in top_results:
            st.write(f"📄 **Document ID:** {doc_id}, 🔢 **Similarity Score:** {score:.4f}")
    else:
        st.warning("Please enter a search query to retrieve documents.")
