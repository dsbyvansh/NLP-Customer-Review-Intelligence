from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def build_centroid_matrix(df,embeddings):
    topic_centroids = {}
    for topic_id in df['topic_id'].unique():
        if topic_id == -1:
            continue
        
        mask = df['topic_id'] == topic_id
        topic_centroids[topic_id] = embeddings[mask].mean(axis=0)
    
    topic_ids_sorted = sorted(topic_centroids.keys())
    centroid_matrix = np.array([topic_centroids[t_id] for t_id in topic_ids_sorted])
    topic_labels_sorted = [df[df['topic_id']==t_id]['topic_label'].iloc[0] for t_id in topic_ids_sorted]

    return topic_centroids,topic_ids_sorted,centroid_matrix,topic_labels_sorted

def predict_proba(texts,model,centroid_matrix):
    embeddings_batch = model.encode(texts) # shape (n,384)
    similarities = cosine_similarity(embeddings_batch,centroid_matrix) # shape (n,39)
    probabilities = softmax(similarities,axis=1) # shape (n,39)
    return probabilities