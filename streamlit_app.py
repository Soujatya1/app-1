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
import time

st.title("Knowledge Management Chatbot")

if not os.path.exists("uploaded_files"):
    os.makedirs("uploaded_files")

# Initialize the interaction history if not present
if 'history' not in st.session_state:
    st.session_state.history = []

# Limit history to the last 5 interactions
def limit_history():
    if len(st.session_state.history) > 5:
        st.session_state.history = st.session_state.history[-5:]

uploaded_files = st.file_uploader("Upload a file", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save each file temporarily in the created directory
        file_path = os.path.join("uploaded_files", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

# Initialize LLM model
llm = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")

# Chat Prompt Template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
<context>
Questions: {input}
"""
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("uploaded_files")
        st.session_state.docs = st.session_state.loader.load()
        st.write(f"Loaded {len(st.session_state.docs)} documents.")
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Display the conversation history at the top
st.header("Conversation History")
for interaction in st.session_state.history:
    st.write(f"**You:** {interaction['question']}")
    st.write(f"**Bot:** {interaction['answer']}")
    st.write("---")

# Divider to separate the conversation history from the input box
st.write("---")
st.write("Ask your question below:")

# Text input for the user to enter the question at the bottom
prompt1 = st.text_input("Enter your question here.....")

# Button to embed documents
if st.button("Embed Docs"):
    vector_embedding()

# Check if the user entered a question and if documents are embedded
if prompt1 and "vectors" in st.session_state:
    # Create a context string that combines the last few interactions
    context = ""
    for interaction in st.session_state.history[-5:]:  # Get last 5 interactions
        context += f"You: {interaction['question']}\nBot: {interaction['answer']}\n"
    
    # Use the context to determine if the new question is related to the previous answer
    is_follow_up = any(prompt1.lower() in follow_up for follow_up in ["elaborate", "tell me more", "can you expand", "in summary", "list points", "give me details"])

    # Prepare the input based on whether it's a follow-up
    if is_follow_up:
        prompt_input = f"Please elaborate on the previous answer: {st.session_state.history[-1]['answer']}"
    else:
        prompt_input = prompt1

    # Create chains for document retrieval and question answering
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure the time to get a response
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt_input, 'context': context})
    st.write("Response time:", time.process_time() - start)

    # Extract the answer from the response
    answer = response['answer']

    # Update the interaction history
    st.session_state.history.append({"question": prompt1, "answer": answer})
    limit_history()  # Limit to the last 5 interactions

    # Display the current answer
    st.write(answer)

    # With a streamlit expander to show the document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
