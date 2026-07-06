import streamlit as st
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer
import pandas as pd 
import numpy as np
import os
from src.classifier import build_centroid_matrix,predict_proba
from src.pipeline import rag_pipeline
from dotenv import load_dotenv

if os.path.exists("../apikey.env"):
    load_dotenv("../apikey.env")

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def load_chroma_collection():
    client = chromadb.PersistentClient(path="data/chromadb")
    collection = client.get_or_create_collection(
        name = "complaint_embeddings",
        metadata= {"hnsw:space":"cosine"}
    )
    return collection

@st.cache_resource
def load_groq_client():
    return Groq(api_key=os.environ.get("GROQ_API_KEY"))

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned/cleaned_reviews.csv")
    embeddings = np.load("data/cleaned/sentence_embeddings.npy")
    return df,embeddings


model = load_model()
collection = load_chroma_collection()
client = load_groq_client()
df,embeddings = load_data()
topic_centroids,topic_ids_sorted,centroid_matrix,topic_labels_sorted = build_centroid_matrix(df,embeddings)

st.title("Customer Support Intelligence system")

st.info("This system is trained on Cell Phones and Accessories complaints",icon="ℹ️")

st.markdown("**Example Complaints**")
if st.button("My Phone battery is not working"):
    st.session_state['complaint'] = "My Phone battery is not working"

if st.button("There is scratch on my screen cover"):
    st.session_state['complaint'] = "There is scratch on my screen cover"

if st.button("Camera lens gets hazed and scratched easily"):
    st.session_state['complaint'] = "Camera lens gets hazed and scratched easily"

complaint = st.text_area("Enter Customer complaint",
                          key='complaint')

if st.button("Analyze"):
    if complaint and complaint.strip():
        probs = predict_proba([complaint],model,centroid_matrix)
        predicted_idx = probs[0].argmax()
        topic_label = topic_labels_sorted[predicted_idx]
        confidence = f"{probs[0][predicted_idx]*100:.1f}%"

        response,docs,used_rag = rag_pipeline(complaint,topic_label,collection,client,model)

        if response is None:
            st.warning("Review is too short ⚠️ Try Again!!")
        else:
            st.markdown("## Complaint Analysis")
            st.markdown("### Complaint topic")
            st.caption("""⚠️ The Confidence score determines how confident the system
            is about the topic label""")
            st.markdown(f"""
            - Topic of Complaint: {topic_label}
            - Confidence Score: {confidence}    
            """)
            st.markdown("### Agent Response")
            st.markdown(f"{response}")
            if docs is not None:
                st.markdown("### Previous Three similar complaints")
                for i in range(len(docs)):
                    st.markdown(f"{i+1}. {docs[i]}")
            
            st.markdown("### Generated Response Type")
            st.markdown("This response is generated using RAG by reviewing the past complaints" if used_rag else "This response is directly generated reviewing your complaint")
            with st.expander("Why this response?"):
                with st.spinner("Generating LIME explanation..."):
                    # explanation here