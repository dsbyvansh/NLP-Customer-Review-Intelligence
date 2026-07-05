from src.retrieval import retrieve_filtered

def generate_response(complaint,topic_label,client,context=None):
    context = context or []
    context_text = "\n".join([f"{i+1}.{c}" for i,c in enumerate(context)])
    if not context:
        prompt = f"Topic: {topic_label}\nComplaint: {complaint}"
    else:
        prompt = f"Topic:{topic_label}\nComplaint: {complaint}\nSimilar past complaints: {context_text}"

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role":"system",
                "content": "You are a helpful customer support agent for an electronics accessories company.Your job is to write a professional, empathetic reply to a customer complaint.Keep your response concise and actionable. Do not include a sign-off, signature, or placeholder like [Your Name]."
            },
            {
                "role":"user",
                "content": prompt
            }
        ],
        model= "llama-3.1-8b-instant"
    )
    return chat_completion.choices[0].message.content

def rag_pipeline(complaint,topic_label,collection,client,model):
    if len(complaint.split())< 5:
        return None
    
    docs,distances = retrieve_filtered(complaint,collection,model,topic_label,top_k=3)

    if distances == [] or distances[0]>0.4:
        response = generate_response(complaint,topic_label,client)
    else:
        response = generate_response(complaint,topic_label,client,context=docs)

    return response