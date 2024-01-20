from openai import OpenAI
import os
from dotenv import load_dotenv
import pymysql
import json 
import spacy
from sentence_transformers import SentenceTransformer

load_dotenv()

OpenAI.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()

# Database configuration
db_host = os.getenv("host")
db_user =  os.getenv("user")
db_password = os.getenv("pass")
db_name = os.getenv("name")

# Establish a connection to the database
connection = pymysql.connect(host=db_host, user=db_user, password=db_password, db=db_name)

def create_table():
    try:
        with connection.cursor() as cursor:
            sql = """
            CREATE TABLE IF NOT EXISTS vectortable (
                id INT AUTO_INCREMENT PRIMARY KEY,
                text TEXT,
                vector JSON
            );
            """
            cursor.execute(sql)
        connection.commit()
    except Exception as e:
        print(f"Error creating table: {e}")


def process_file(file_path) : 
    with open(file_path, 'r', encoding='utf-8') as file:

        text = file.read()

        nlp = spacy.load("en_core_web_sm")

        doc = nlp(text)

        sentences = [sent.text.strip() for sent in doc.sents]

        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

        embeddings = sbert_model.encode(sentences)

        embeddings_json = [json.dumps(embedding.tolist()) for embedding in embeddings]

        for sentence, embedding_str in zip(sentences, embeddings_json):
            with connection.cursor() as cursor:
                sql = "INSERT INTO vectortable (text, vector) VALUES (%s,%s);"
                cursor.execute(sql, (sentence,embedding_str))
                connection.commit()


create_table()
process_file('guidelinescopy.txt')
connection.close()
