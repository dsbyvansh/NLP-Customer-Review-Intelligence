import streamlit as st
from sentence_transformers import SentenceTransformer 
import numpy as np
from src.classifier import build_centroid_matrix,predict_proba
from src.pipeline import rag_pipeline
import matplotlib.pyplot as plt
from src.loader import load_chroma_collection,load_model,load_groq_client,load_data

st.set_page_config(page_title="Customer Support Intelligence system",layout="wide",page_icon="❗")  

with st.spinner("Loading models... this may take a moment on first run⏳"):
    model = load_model()
    collection = load_chroma_collection()
    client = load_groq_client()
    df,embeddings = load_data()
    topic_centroids,topic_ids_sorted,centroid_matrix,topic_labels_sorted = build_centroid_matrix(df,embeddings)

with st.sidebar:
    st.markdown("### About this tool")
    st.info("""
    This system uses Natural Language Processing 
    which is trained on 30K+ Amazon reviews for the  
    category "Cell Phones and Accessories" for generating
    customer facing responses
    """)
    
    st.markdown("### How it works")
    st.markdown("""
    1. Enter a complaint query for the category "Cell Phones and accessories
       (Example comlaints are provided)
    2. Click **Analyze**
    3. View generated response and similar past complaints            
    """)
    st.markdown("""## Pipeline

        Complaint

            ↓

        Topic Classification

            ↓

        Response Generation

            ↓

        Similar Complaints

            ↓

        LIME Explainability""")

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
    with st.spinner("Analyzing..."):
        if complaint and complaint.strip():
            probs = predict_proba([complaint],model,centroid_matrix)
            predicted_idx = probs[0].argmax()
            topic_label = topic_labels_sorted[predicted_idx]
            sorted_probs = np.sort(probs[0])[::-1]
            margin = sorted_probs[0] - sorted_probs[1]
            if margin>0.005:
                confidence = "🟢 High"
            elif margin>0.002:
                confidence = "🟡 Medium"
            else:
                confidence = "🔴 Low"

            response,docs,used_rag = rag_pipeline(complaint,topic_label,collection,client,model)

            if response is None:
                st.warning("Review is too short ⚠️ Try Again!!")
            else:
                st.session_state['topic_label'] = topic_label
                st.session_state['confidence'] = confidence
                st.session_state['predicted_idx'] = predicted_idx
                st.session_state['response'] = response
                st.session_state['docs'] = docs
                st.session_state['used_rag'] = used_rag
                st.session_state['input_complaint'] = complaint
                st.switch_page("pages/result.py")