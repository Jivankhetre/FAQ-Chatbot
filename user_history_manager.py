from google.cloud import storage
from datetime import datetime
import json
SERVICE_ACCOUNT_FILE = "./gemini_cred.json"

def store_user_history(bucket_name, user_id, session_id, query, response):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_path = f"user_history/{user_id}/{session_id}.json"
    blob = bucket.blob(blob_path)
    
    existing_history = {}
    if blob.exists():
        existing_history = json.loads(blob.download_as_text())
    
    interaction = {
        "query": query,
        "response": response,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if "interactions" not in existing_history:
        existing_history["interactions"] = []
    existing_history["interactions"].append(interaction)
    
    data_json = json.dumps(existing_history)
    blob.upload_from_string(data_json, content_type='application/json')

def get_user_history(bucket_name, user_id, session_id):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob_path = f"user_history/{user_id}/{session_id}.json"
    blob = bucket.blob(blob_path)
    
    if not blob.exists():
        return None
    
    data_json = blob.download_as_text()
    return json.loads(data_json)
