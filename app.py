from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from google.cloud import storage
import faiss
import json
import os
from data_loader import load_faiss_index, load_documents, initialize_vertex_ai
from similarity_search import get_most_similar_document
from response_generator import generate_response
from user_history_manager import store_user_history, get_user_history

# Debug print statement to verify the environment variable
print(f"GOOGLE_APPLICATION_CREDENTIALS: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")

# Initialize FastAPI app
app = FastAPI()

# Initialize Vertex AI
initialize_vertex_ai()

# Load the FAISS index and documents
faiss_index = load_faiss_index("./faiss_index.index")
all_documents = load_documents("./all_documents.json")

# Chatbot prompt
Chat = """
You are a highly knowledgeable and professional FAQ specialist who will receive:
A specific query from a user.
A RAG document containing relevant information and data.
Your task is to answer the user's query by searching for relevant information within the RAG document. You must cross-verify the information to ensure accuracy.
Consider the user's query and the information from the document before answering. Your response should be detailed and comprehensive, providing additional context and related information where appropriate. Aim to make your response sound more human-like and engaging. Include explanations, examples, or additional details that may help the user understand the topic better.
If the relevant information is not found in the document, or if the query is unclear, you must simply answer "My knowledge base doesn't include that information, please ask a different query, or reword your current query".
Only answer the query if you have all the necessary information, and when you do answer, be detailed and comprehensive. Provide as much relevant information from the document as possible.
Your response should strictly follow the following format:
Response: The answer to the query
Reference: Document used as reference for the answer
GCS URI: The Google Cloud Storage URI of the document used as reference.
For example, if the document containing the relevant information is named the-basics-of-making-a-will and is located in the rag-test2 bucket, the GCS URI should be formatted as follows:
GCS URI: gs://rag-test2/the-basics-of-making-a-will
Important Notes:
Always identify the specific document that contains the relevant information for each query.
Ensure that the GCS URI provided is accurate and directly corresponds to the document used to answer the query.
Do not use a generic or incorrect URI. Each response must accurately reflect the document that was used to answer the query.
If multiple documents are relevant, cite the most specific document that directly answers the query.
Extract the filename from the provided GCS URI and replace the bucket name with rag-test2. For example, if the provided GCS URI is gs://asd-in/faqs-categories/the-basics-of-making-a-will, the correct GCS URI should be gs://rag-test2/the-basics-of-making-a-will.
The filename should be the last part of the provided GCS URI after the last /. Do not include any additional context or path information.
Ensure that the filename is not URL-encoded. For example, if the filename is Will FAQs, it should be used as is without any encoding.
If the provided GCS URI includes any path information, ignore it and use only the filename.
The filename should be the exact name of the document from which the response is taken. Do not include any additional text or context.
[Provide the direct citation links here. These are ONLY the google cloud storage bucket link from the info section. No other links from any other websites should be provided.]
Example Response:
Response: To create a Yellow Will, you must be at least 18 years old. This age requirement ensures that you have the legal capacity to make decisions about your assets and affairs. It's important to note that creating a will is a significant step in estate planning, allowing you to specify how your assets should be distributed after your passing. Additionally, having a will can help reduce potential disputes among family members and ensure that your wishes are carried out.
Reference: the-basics-of-making-a-will
GCS URI: gs://rag-test2/the-basics-of-making-a-will
"""

# Define request model for query
class QueryRequest(BaseModel):
    query: str
    user_id: str
    session_id: str

# Define request model for ending session
class EndSessionRequest(BaseModel):
    user_id: str
    session_id: str

# Define response model
class QueryResponse(BaseModel):
    response: str
    reference: str
    gcs_uri: str

# Define request model for Dialogflow CX webhook
class DialogflowRequest(BaseModel):
    sessionInfo: dict
    fulfillmentInfo: dict

# Define response model for Dialogflow CX webhook
class DialogflowResponse(BaseModel):
    fulfillmentResponse: dict

# In-memory storage for session data
session_data = {}

@app.post("/query", response_model=QueryResponse)
async def handle_query(request: QueryRequest):
    query = request.query
    user_id = request.user_id
    session_id = request.session_id

    # Get the most similar document
    context, gcs_uri = get_most_similar_document(query, faiss_index, all_documents)

    if context is None or gcs_uri is None:
        raise HTTPException(status_code=404, detail="Failed to retrieve context or GCS URI.")

    # Prepare the prompt
    prompt = f'''Context: {context}
    GCS URI: {gcs_uri}
    Query: {query}'''

    # Placeholder for safety settings and generation config
    safety_settings = {}
    generation_config = {}
    system_instructions = Chat

    # Generate response
    response = generate_response(prompt, context, gcs_uri, system_instructions, safety_settings, generation_config)

    # Store query and response in session data
    if session_id not in session_data:
        session_data[session_id] = []
    session_data[session_id].append({
        "query": query,
        "response": response
    })

    # Return the response
    return QueryResponse(response=response, reference=context, gcs_uri=gcs_uri)

@app.post("/end_session")
async def end_session(request: EndSessionRequest):
    user_id = request.user_id
    session_id = request.session_id

    # Get the session data
    session_queries = session_data.get(session_id, [])

    # Store user history
    bucket_name = "rag-test2"
    for interaction in session_queries:
        store_user_history(bucket_name, user_id, session_id, interaction["query"], interaction["response"])

    # Clear session data
    if session_id in session_data:
        del session_data[session_id]

    return {"message": "Session ended and history updated"}
@app.post("/dialogflow_webhook", response_model=DialogflowResponse)
async def dialogflow_webhook(request: Request):
    payload = await request.json()
    print(payload)
    query = payload.get("sessionInfo", {}).get("parameters", {}).get("query")
    user_id = payload.get("sessionInfo", {}).get("parameters", {}).get("user_id")
    print(user_id)
    session_id = payload.get("sessionInfo", {}).get("parameters", {}).get("session_id")
    tag = payload.get("fulfillmentInfo", {}).get("tag")

    if not query or not user_id or not session_id:
        raise HTTPException(status_code=400, detail="Missing required parameters")

    # Handle different tags
    if tag == "generate_will":
        # Specific logic for generating a will
        context, gcs_uri = get_most_similar_document(query, faiss_index, all_documents)

        if context is None or gcs_uri is None:
            raise HTTPException(status_code=404, detail="Failed to retrieve context or GCS URI.")

        # Prepare the prompt
        prompt = f'''Context: {context}
        GCS URI: {gcs_uri}
        Query: {query}'''

        # Placeholder for safety settings and generation config
        safety_settings = {}
        generation_config = {}
        system_instructions = Chat

        # Generate response
        response = generate_response(prompt, context, gcs_uri, system_instructions, safety_settings, generation_config)
    else:
        # Default logic
        response = "My knowledge base doesn't include that information, please ask a different query, or reword your current query."

    # Store query and response in session data
    if session_id not in session_data:
        session_data[session_id] = []
    session_data[session_id].append({
        "query": query,
        "response": response
    })

    # Prepare Dialogflow CX response
    dialogflow_response = {
        "fulfillmentResponse": {
            "messages": [
                {
                    "text": {
                        "text": [response]
                    }
                }
            ]
        }
    }

    return DialogflowResponse(**dialogflow_response)
# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
