import streamlit as st
import sys

# Use pysqlite3 to override sqlite3
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import WebBaseLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import GooglePalmEmbeddings  # Updated import
import os

# Set Google API key
os.environ['GOOGLE_API_KEY'] = st.secrets["GOOGLE_API_KEY"]

def load_website(url):
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        return data
    except Exception as e:
        st.error(f"Failed to load website: {e}")
        return []

# Streamlit UI
st.title("Scrappy-Doo")
st.write("Extract data from any website with ease!")

link = st.text_input("URL:")

if st.button("Fetch"):
    if link:
        with st.spinner("Loading website data..."):
            website_data = load_website(link)
            if website_data:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50,
                    length_function=len
                )
                split_docs = text_splitter.split_documents(website_data)
                embed_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                try:
                    vectorstore = Chroma.from_documents(split_docs, embed_model)
                except Exception as e:
                    st.error(f"Failed to initialize Chroma: {e}")
                    st.stop()
                llm = GoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                qa = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=vectorstore.as_retriever(),
                    memory=memory
                )
                st.session_state.qa = qa
                st.session_state.vectorstore = vectorstore
                st.success("Website data loaded successfully! You can now ask questions.")
    else:
        st.warning("Please enter a URL before clicking Fetch!")

if "qa" in st.session_state:
    query = st.text_input("Ask a question about the website:")
    if st.button("Ask"):
        if query:
            try:
                response = st.session_state.qa({"question": query})
                st.subheader("AI Response:")
                st.write(response.get('answer', response))
            except Exception as e:
                st.error(f"Error processing query: {e}")
        else:
            st.warning("Please enter a question before clicking Ask!")
else:
    st.warning("Please fetch a URL before asking questions.")
