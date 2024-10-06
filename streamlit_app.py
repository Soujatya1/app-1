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

# Set the title of the app
st.title("Knowledge Management Chatbot")

# Create a directory for uploaded files if it doesn't exist
if not os.path.exists("uploaded_files"):
    os.makedirs("uploaded_files")

# Initialize the interaction history and context if not present
if 'history' not in st.session_state:
    st.session_state.history = []
if 'context' not in st.session_state:
    st.session_state.context = ""

# File uploader for PDF documents
uploaded_files = st.file_uploader("Upload a file", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save each file temporarily in the created directory
        file_path = os.path.join("uploaded_files", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

# Initialize LLM model
llm = ChatGroq(groq_api_key="YOUR_API_KEY", model_name="Llama3-8b-8192")

# Define the prompt template for the chatbot
prompt_template = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Function to embed documents
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

# If a question is entered and documents are embedded
if prompt1 and "vectors" in st.session_state:
    # Get the last question and answer for context
    last_question = st.session_state.history[-1]["question"] if st.session_state.history else ""
    print(f"Last Question: {last_question}")  # Debugging output

    # Check if the last question and current question are related
    is_related = last_question.lower() in prompt1.lower() or prompt1.lower() in last_question.lower()
    if is_related:
        current_context = st.session_state.context  # Use existing context
    else:
        current_context = ""  # Reset context

    # Create chains for document retrieval and question answering
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure the time to get a response
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1, 'context': current_context})
    st.write("Response time :", time.process_time() - start)

    # Extract the answer from the response
    answer = response.get('answer', "Sorry, I couldn't find an answer.")
    print(f"Answer: {answer}")  # Debugging output

    # Update context based on current question
    st.session_state.context = prompt1 if not is_related else st.session_state.context

    # Append the interaction to the session state history
    st.session_state.history.append({"question": prompt1, "answer": answer})

    # Display the current answer
    st.write(answer)

    # Show the document similarity search results in an expander
    with st.expander("Document Similarity Search"):
        for doc in response.get("context", []):
            st.write(doc.page_content)
            st.write("--------------------------
