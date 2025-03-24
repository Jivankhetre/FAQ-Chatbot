import numpy as np
from langchain_google_vertexai import VertexAIEmbeddings

def get_most_similar_document(query, faiss_index, all_documents):
    embeddings = VertexAIEmbeddings(model_name="text-embedding-004")
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array([query_embedding]).astype('float32')
    
    D, I = faiss_index.search(query_embedding, k=1)
    most_similar_document = all_documents[I[0][0]]
    
    if 'metadata' not in most_similar_document or 'gcs_uri' not in most_similar_document['metadata']:
        print("Metadata or GCS URI is missing in the document.")
        return None, None
    
    output = most_similar_document['page_content']
    gcs_uri = most_similar_document['metadata']['gcs_uri']
    return output, gcs_uri