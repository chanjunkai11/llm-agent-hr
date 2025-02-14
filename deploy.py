import sys
sys.modules["sqlite3"] = __import__("pysqlite3")
import chromadb
import streamlit as st
import os
from langchain.schema import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import psycopg2
import re


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


st.title("RAG Chatbot")

vectorstore = load_chroma_index()
llm = ChatGoogleGenerativeAI(model=chat_model_name)

query = st.text_input("Ask a question:")

if st.button("Generate Response"):
    if query:
        docs1 = vectorstore[1].query(
            query_texts=[query],
            n_results=5,  # Number of similar documents to retrieve
            include=["documents", "distances"]  # Ensure distances are included in the results
        )
        context1 = "\n\n".join(docs1['documents'][0])

        if not docs1:
            st.warning("No relevant documents found in the vector store. Try another query.")
        else:
            # Construct the LLM message format
            messages = [
                SystemMessage(content="You are a posgreSQL Interpreter."),
                HumanMessage(
                    content=f"""Given the following postgreSQL schema:

                    {context1}

                    Determine where the user's query is able to convert into postgreSQL statement:

                    {query}
                    
                    output then classify as SQL if can convert into postgreSQL statement else is not SQL.
                    no description or yapping is needed just the class value is needed. Your response should only be SQL or not SQL not both.
                    """
                ),
            ]

            # Get response from LLM
            response = llm.invoke(messages)

        if response.content == 'SQL':
            if not docs1:
                st.warning("No relevant documents found in the vector store. Try another query.")
            else:
                # Construct the LLM message format
                messages = [
                    SystemMessage(content="You are a posgreSQL SQL statement generator."),
                    HumanMessage(
                        content=f"""You will help users translate their input natural language query requirements into postgreSQL SQL statements that can be process by psycopg2.
                        
                        {query}

                        Depending on user's input, you may need to use 
                        
                        {context1} 
                        
                        to perform joining or other information to generate the correct sql. Please take note that you must know which schema to use from. 
                        no other description or yapping just give the sql statement and nothing else 
                        """
                    ),
                ]

                # Get response from LLM
                response1 = llm.invoke(messages)
                st.write(response1.content)
                # Display response
                st.write("### Answer:")
                sql_pattern = r"```sql\n(.*?)\n```"
                match = re.search(sql_pattern, response1.content, re.DOTALL)
                if match:
                    # Strip any leading/trailing whitespace and return the SQL statement
                    sql_statement = match.group(1).strip()

                    try:
                        conn = psycopg2.connect(postgres_ip)
                        cursor = conn.cursor()
                        cursor.execute(sql_statement)
                        rows = cursor.fetchall()

                        # Construct the LLM message format
                        messages = [
                            SystemMessage(content="You are a helpful assistant."),
                            HumanMessage(
                                content=f"""Given the following data acquired from database and the sql statement:

                                {rows}

                                Answer the user's query:

                                {query}

                                display and describe out the results in a organized manner. Do not show SQL statement out.
                                """
                            ),
                        ]

                        # Get response from LLM
                        response2 = llm.invoke(messages)
                        # Print results
                        st.write(response2.content)

                        # Close cursor and connection
                        cursor.close()
                        conn.close()
                    except Exception as e:
                        st.write(e)
                    
                else:
                    st.write("No SQL provided")
        else:
            # Retrieve relevant documents
            docs = vectorstore[0].query(
                query_texts=[query],
                n_results=7,  # Number of similar documents to retrieve
                include=["documents", "distances"]  # Ensure distances are included in the results
            )
            context = "\n\n".join(docs['documents'][0])

            if not docs:
                st.warning("No relevant documents found in the vector store. Try another query.")
            else:
                # Construct the LLM message format
                messages = [
                    SystemMessage(content="You are a HR assistant."),
                    HumanMessage(
                        content=f"""Given the following documents:

                        {context}

                        Answer the user's query:

                        {query}

                        Accurately, and do not provide any information that is not included in the documents.
                        """
                    ),
                ]

                # Get response from LLM
                response3 = llm.invoke(messages)

                # Display response
                st.write("### Answer:")

                # Display cleaned response
                st.write(response3.content)
    else:
        st.warning("Please enter a question.")
