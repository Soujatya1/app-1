import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup

# Streamlit UI
st.title("Website Intelligence")

api_key = "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"
llm = ChatGroq(groq_api_key=api_key, model_name='llama-3.1-70b-versatile', temperature=0.2, top_p=0.2)
hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Input Sitemap URLs and Filters
sitemap_urls_input = "https://www.reliancenipponlife.com/sitemap.xml\nhttps://www.hdfclife.com/universal-sitemap.xml"
filter_words_input = "retirement-plans"

sitemap_urls = sitemap_urls_input.splitlines()
filter_urls = filter_words_input.splitlines()

def load_documents_from_url(url):
    """Loads documents from a given URL."""
    try:
        loader = WebBaseLoader(web_path=url)
        return loader.load()
    except Exception as e:
        st.write(f"Error loading {url}: {e}")
        return []

# Define Cached Functions
@st.cache_data
def load_and_split_documents(urls, filters):
    """Loads and processes documents from sitemaps."""
    loaded_docs = []
    for sitemap_url in urls:
        try:
            response = requests.get(sitemap_url)
            sitemap_content = response.content
            soup = BeautifulSoup(sitemap_content, 'xml')
            urls = [loc.text for loc in soup.find_all('loc')]
            selected_urls = [url for url in urls if any(filter in url for filter in filters)]
            
            for url in selected_urls:
                docs = load_documents_from_url(url)
                for doc in docs:
                    doc.metadata["source"] = url
                loaded_docs.extend(docs)
        except Exception as e:
            st.write(f"Error processing sitemap {sitemap_url}: {e}")
    return loaded_docs
    
def create_embeddings(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    vector_db = FAISS.from_documents(chunks, hf_embedding)
    return vector_db, chunks

# Load and Process Documents with Caching
with st.spinner("Loading and processing documents..."):
    loaded_docs = load_and_split_documents(sitemap_urls, filter_urls)

st.write(f"Loaded documents: {len(loaded_docs)}")

# Create Embeddings with Caching
with st.spinner("Generating embeddings and creating vector database..."):
    vector_db, document_chunks = create_embeddings(loaded_docs)

st.write(f"Number of chunks: {len(document_chunks)}")

# Craft ChatPrompt Template
prompt = ChatPromptTemplate.from_template(
    """
        You are a Life Insurance specialist who needs to answer queries based on the information provided in the websites only. Please follow all the websites, and answer as per the same.

        Do not answer anything except from the website information which has been entered. Please do not skip any information from the tabular data in the website.

        Do not skip any information from the context. Answer appropriately as per the query asked.

        Now, being an excellent Life Insurance agent, you need to compare your policies against the other company's policies in the websites, if asked.

        Generate tabular data wherever required to classify the difference between different parameters of policies.

        I will tip you with a $1000 if the answer provided is helpful.

        <context>
        {context}
        </context>

        Question: {input}"""
)

# Stuff Document Chain Creation
document_chain = create_stuff_documents_chain(llm, prompt)

# Retriever from Vector store
retriever = vector_db.as_retriever()

# Create a retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Query Input Box
with st.container():
    prompt1 = st.text_input("Enter your question here.....", key="user_input", placeholder="Type your question...")

# Query Processing
if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever(search_type="similarity", k=2)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    response = retrieval_chain.invoke({'input': prompt1})
    answer = response['answer']
    st.write("Response:")
    st.write(answer)
