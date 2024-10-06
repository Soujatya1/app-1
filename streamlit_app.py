import streamlit as st
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.schema import HumanMessage, AIMessage  # Importing message classes
import os
import time

st.title("Knowledge Management Chatbot")

# Initialize the interaction history if not present
if 'history' not in st.session_state:
    st.session_state.history = []

# Initialize flowmessages if not present
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = []

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
prompt_template = ChatPromptTemplate.from_template(
"""
The following is a conversation history between a user and a bot.
Use the context of the conversation history to respond to the user's question accurately.

CONVERSATION HISTORY:
{history}

NEW QUESTION: {input}

Please generate a relevant response based on the conversation.
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

# Function to form context from the last 10 interactions
def get_last_context():
    history = st.session_state.history[-10:]  # Take the last 10 interactions
    context = "\n".join([f"User: {h['question']}\nBot: {h['answer']}" for h in history])
    return context

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

# Function to get a response from the chat model
def get_chatmodel_response(question):
    # Prepare the context
    context = get_last_context()
    
    # Create input for the prompt
    prompt_input = prompt_template.format(history=context, input=question)
    
    # Measure the time to get a response
    start = time.process_time()
    
    # Call the model with the formatted prompt
    response = llm.invoke(prompt_input)  # Pass the formatted string directly
    
    st.write("Response time :", time.process_time() - start)
    
    # Extract the answer from the response
    answer = response['answer']
    
    # Append user message to flowmessages
    st.session_state['flowmessages'].append(HumanMessage(content=question))
    st.session_state['flowmessages'].append(AIMessage(content=answer))
    
    # Append the interaction to the session state history
    st.session_state.history.append({"question": question, "answer": answer})
    
    return answer

# If a question is entered and documents are embedded
if prompt1 and "vectors" in st.session_state:
    # Get response from the chat model
    answer = get_chatmodel_response(prompt1)
    
    # Display the current answer
    st.write(answer)
    
    # With a streamlit expander to show the document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response.get("context", [])):
            st.write(doc.page_content)
            st.write("--------------------------------")
