import streamlit as st
from lime.lime_text import LimeTextExplainer
import matplotlib.pyplot as plt
from src.loader import load_chroma_collection,load_model,load_groq_client,load_data
from src.classifier import build_centroid_matrix,predict_proba

st.set_page_config(page_title="Customer Support Intelligence system",layout="wide") 

if 'response' not in st.session_state:
    st.warning("Please Enter a complaint first")
    if st.button("Go Back"):
        st.switch_page("app.py")
    st.stop()

with st.spinner("Loading models... this may take a moment on first run⏳"):
    model = load_model()
    collection = load_chroma_collection()
    client = load_groq_client()
    df,embeddings = load_data()
    topic_centroids,topic_ids_sorted,centroid_matrix,topic_labels_sorted = build_centroid_matrix(df,embeddings)

topic_label = st.session_state['topic_label']
confidence = st.session_state['confidence']
predicted_idx = st.session_state['predicted_idx']
response = st.session_state['response']
docs = st.session_state['docs']
used_rag = st.session_state['used_rag']
complaint = st.session_state['input_complaint']


st.markdown("## Complaint Analysis")
st.markdown("### Complaint topic")
st.caption("""⚠️ The Confidence score determines how confident the system
            is about the topic label""")
st.markdown(f"""
    - Topic of Complaint: {topic_label}
    - Confidence: {confidence}    
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
        lime_predict = lambda texts: predict_proba(texts,model,centroid_matrix)
        explainer = LimeTextExplainer(
                    class_names=topic_labels_sorted,
                    random_state=42
                    )

        explanation = explainer.explain_instance(
        text_instance=complaint,
        classifier_fn=lime_predict,
        num_features=10,
        num_samples=500,
        labels=[predicted_idx]
        )

    word_weights = explanation.as_list(label=predicted_idx)

    words = [w[0] for w in word_weights]
    weights = [w[1] for w in word_weights]
    colors = ['green' if w>0 else 'red' for w in weights]

    
    fig,ax = plt.subplots(  )
    ax.barh(words,weights,color=colors)
    ax.set_ylabel("Words")
    ax.set_xlabel("Weights")
    ax.set_title(f"LIME explanation - {topic_label}")
    st.caption("""🟢 Green bars = words that pushed the prediction TOWARD this topic
    🔴 Red bars = words that pushed the prediction AWAY from this topic
    Longer bar = stronger influence on the topic classification.""")
    st.pyplot(fig)

if st.button("Analyze another compaint"):
    st.switch_page("app.py")    