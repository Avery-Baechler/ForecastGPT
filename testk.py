import pymysql
import numpy as np
from scipy.spatial.distance import cosine
import json
from openai import OpenAI
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


OpenAI.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

connection = pymysql.connect(
        host=os.getenv("host")
        user=os.getenv("user"),
        password=os.getenv("pass"),
        db=os.getenv("name") 
        )


def fetch_vectors():
    vectors = []
    with connection.cursor() as cursor:
        cursor.execute("SELECT id, vector FROM vectortable")
        for row in cursor.fetchall():
            id, vector = row
            vectors.append((id, np.array(json.loads(vector))))
    return vectors

def create_embed(text):
    # =============== For GPT Embedings ================
    #response = client.embeddings.create(
    #    input=text,
    #    model="text-embedding-ada-002"
    #)
    #embedding = response.data[0].embedding
    #return np.array(embedding)

    #================ SBERT Embedings ================
    sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding =  sbert_model.encode([query])[0]
    return np.array(embedding)

def cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)

def get_text_by_id(id):
    with connection.cursor() as cursor:
        cursor.execute("SELECT text FROM vectortable WHERE id = %s", (id,))
        result = cursor.fetchone()
        if result:
            return result[0]
        else:
            return None

def top_k_search(query, k=5):
    query_vector = create_embed(query)
    fetched_vectors = fetch_vectors()
    similarities = [(id, cosine_similarity(query_vector, vector)) for id, vector in fetched_vectors]
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_k = similarities[:k]

    top_k_texts = [(id, similarity, get_text_by_id(id)) for id, similarity in top_k]
    return top_k_texts

query = "recent snow fall effect on creating snow slabs" 
create_embed(query)
top_k_results = top_k_search(query, k=10)
for result in top_k_results:
    print(f"ID: {result[0]}, Similarity: {result[1]}, Text: {result[2]}")

