import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from groq import Groq
import pandas as pd
import numpy as np
import os
from load_dotenv import load_dotenv 

if os.path.exists("apikey.env"):
    load_dotenv("apikey.env")

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