import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os

uploaded_files = st.file_uploader("Upload a file", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_documents = []

    # Create a directory to store uploaded files temporarily
    temp_dir = "uploaded_files"
    os.makedirs(temp_dir, exist_ok=True)

    # Iterate over each uploaded file
    for uploaded_file in uploaded_files:
        # Save each file temporarily
        with open(os.path.join(temp_dir, uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Success message
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

st.title("Knowledge Management Chatbot")

llm=ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")

prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

def vector_embedding():
    

    if "vectors" not in st.session_state:

        st.session_state.embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader=PyPDFDirectoryLoader("uploaded_file.name")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

prompt1=st.text_input("Enter your question here.....")


if st.button("Embed Docs"):
    vector_embedding()

import time



if prompt1:
    document_chain=create_stuff_documents_chain(llm,prompt)
    retriever=st.session_state.vectors.as_retriever()
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    start=time.process_time()
    response=retrieval_chain.invoke({'input':prompt1})
    st.write("Response time :",time.process_time()-start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
