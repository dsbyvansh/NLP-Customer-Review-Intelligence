from src.preprocessing import text_preprocessing

def retrieve_filtered(query,collection,model,topic_label,top_k=3):
    query = text_preprocessing(query,remove_stopwords=False)
    query_embedding = model.encode(query).tolist()
    result = collection.query(
        query_embeddings = [query_embedding],
        n_results=top_k,
        include= ['documents','metadatas','distances'],
        where= {"topic_label":{"$eq": topic_label}}
    )
    return result['documents'][0],result['distances'][0]    