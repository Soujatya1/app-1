import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from langchain.chains import LLMChain, StuffDocumentsChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import LLMChain, StuffDocumentsChain
import os
import time

st.title("Knowledge Management Chatbot")

# Initialize the interaction history if not present
if 'history' not in st.session_state:
    st.session_state.history = []

# Initialize flowmessages if not present
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [SystemMessage(content="You are a document-based AI assistant")]

# Upload PDF files
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
Use the following context from the uploaded documents for answering the question:
<context>
{context}
</context>
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

# Function to form context from the last 10 interactions only if relevant
def get_last_context(question):
    last_context = ""
    history = st.session_state.history[-10:]  # Take the last 10 interactions
    for h in history:
        # If the new question is related to the previous one (heuristically)
        if question in h['question'] or h['question'] in question:
            last_context += f"Q: {h['question']}\nA: {h['answer']}\n"
    return last_context

# Function to get AI response based strictly on documents
# Function to get AI response based strictly on documents
def get_chatmodel_response_from_docs(question, context):
    # Create the document retrieval chain using the LLM and prompt
    document_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    
    # Use the FAISS vector store to retrieve relevant documents
    relevant_docs = st.session_state.vectors.similarity_search(question)
    
    # Ensure relevant_docs is a list of documents
    if not relevant_docs:
        return "No relevant documents found."

    # Execute the chain
    response = document_chain.invoke({
        "input_documents": relevant_docs,
        "input": question,
        "context": context,
    })

    # Ensure that the response is strictly from the document
    return response['output']  # Adjust this according to the response structure you expect


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
    # Retrieve context based on whether the new question relates to past history
    context = get_last_context(prompt1)
    
    # Measure the time to get a response
    start = time.process_time()
    
    # Get the answer strictly from the documents
    answer = get_chatmodel_response_from_docs(prompt1, context)
    
    # Display the response time
    st.write("Response time:", time.process_time() - start)
    
    # Append the interaction to the session state history
    st.session_state.history.append({"question": prompt1, "answer": answer})
    
    # Display the current answer
    st.write(answer)
    
    # With a Streamlit expander to show the document similarity search results
    with st.expander("Document Similarity Search"):
        # Retrieve the relevant documents based on similarity
        response = st.session_state.vectors.similarity_search_with_score(prompt1)
        for i, (doc, score) in enumerate(response):
            st.write(f"Document {i + 1} (Score: {score}):")
            st.write(doc.page_content)
            st.write("--------------------------------")
