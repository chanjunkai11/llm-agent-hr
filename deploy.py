import sys
sys.modules["sqlite3"] = __import__("pysqlite3")
import chromadb
import streamlit as st
import os
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import psycopg2
import re

st.title("RAG Chatbot")

db_secrets = st.secrets["database"]
api_secrets = st.secrets["api"]

postgres_ip = db_secrets["postgres_ip"]
os.environ["GOOGLE_API_KEY"] = api_secrets["gemini_api"]  # Set your Gemini API key here
CHROMA_DB_DIR = "./chroma_db"
embedding_model_name = "models/text-embedding-004"  # Gemini Embeddings
chat_model_name = "gemini-2.0-flash"  # Gemini Chat Model

class CustomEmbeddingFunction:
    def __init__(self, model_name):
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=model_name)

    def __call__(self, input):
        return self.embedding_model.embed_documents(input)

def get_embedding_function(model_name):
    embeddings = CustomEmbeddingFunction(model_name)
    return embeddings

def load_chroma_index():
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
    
    # Load the stored collection
    collection = chroma_client.get_collection("my_collection", embedding_function=get_embedding_function(embedding_model_name))   
    collection1 = chroma_client.get_collection("my_database", embedding_function=get_embedding_function(embedding_model_name))   
    return collection, collection1

client = ChatGoogleGenerativeAI(model=chat_model_name)
vectorstore = load_chroma_index()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if query := st.chat_input("Ask a question:"):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    
    with st.chat_message("assistant"):
        docs1 = vectorstore[1].query(
            query_texts=[query],
            n_results=5,
            include=["documents", "distances"]
        )
        context1 = "\n\n".join(docs1['documents'][0]) if docs1 else ""

        if docs1:
            messages = [
                SystemMessage(content="You are a PostgreSQL Interpreter."),
                HumanMessage(
                    content=f"""Given the following PostgreSQL schema:
                    
                    {context1}
                    
                    Determine whether the user's query can be converted into a PostgreSQL statement:
                    
                    {query}
                    
                    Output should be either 'SQL' if convertible or 'not SQL' if not. No extra text.
                    """
                ),
            ]
            response = client.invoke(messages)
        
        if response.content == 'SQL':
            messages = [
                SystemMessage(content="You are a PostgreSQL SQL statement generator."),
                HumanMessage(
                    content=f"""Translate the user's query into a PostgreSQL SQL statement using the schema:
                    
                    {context1}
                    
                    {query}
                    
                    Ensure compatibility with psycopg2. Output only the SQL statement, no extra text.
                    """
                ),
            ]
            response1 = client.invoke(messages)
            sql_pattern = r"```sql\n(.*?)\n```"
            match = re.search(sql_pattern, response1.content, re.DOTALL)
            
            if match:
                sql_statement = match.group(1).strip()
                try:
                    conn = psycopg2.connect(postgres_ip)
                    cursor = conn.cursor()
                    cursor.execute(sql_statement)
                    rows = cursor.fetchall()
                    cursor.close()
                    conn.close()

                    messages = [
                        SystemMessage(content="You are a helpful assistant."),
                        HumanMessage(
                            content=f"""Given the retrieved database results:
                            
                            {rows}
                            
                            Answer the user's query:
                            
                            {query}
                            
                            Present the results in an organized manner without showing the SQL statement.
                            """
                        ),
                    ]
                    response2 = client.invoke(messages)
                    st.markdown(response2.content)
                except Exception as e:
                    st.write(e)
            else:
                st.write("No valid SQL provided.")
        else:
            docs = vectorstore[0].query(
                query_texts=[query],
                n_results=7,
                include=["documents", "distances"]
            )
            context = "\n\n".join(docs['documents'][0]) if docs else ""
            
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(
                    content=f"""Given the following documents:
                    
                    {context}
                    
                    Answer the user's query:
                    
                    {query}
                    
                    Ensure responses are accurate and based only on the documents.
                    """
                ),
            ]
            response3 = client.invoke(messages)
            st.markdown(response3.content)
    
    st.session_state.messages.append({"role": "assistant", "content": response3.content if response.content != 'SQL' else response2.content})
