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
from langchain.schema import SystemMessage, HumanMessage, AIMessage

st.title("Knowledge Management Chatbot")

# Initialize the interaction history if not present
if 'history' not in st.session_state:
    st.session_state.history = []

# Initialize the flow messages if not present
if 'flowmessages' not in st.session_state:
    st.session_state['flowmessages'] = [
        SystemMessage(content="You are a Knowledge Management Specialist who answers questions based on the documents uploaded")
    ]

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
chat_prompt = ChatPromptTemplate.from_template(
"""
Use the following context for answering the question:
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

# Function to get the last context from history
def get_last_context():
    history = st.session_state.history[-10:]  # Take the last 10 interactions
    context = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history])
    return context

# Function to get the chat model response
def get_chatmodel_response(question):
    # Add the user's question to the flow messages
    st.session_state['flowmessages'].append(HumanMessage(content=question))

    # Retrieve the last context from the session state
    context = get_last_context()

    # Create the prompt using the ChatPromptTemplate
    prompt = chat_prompt.format(context=context, input=question)

    # Call the model with the prompt as a string
    response = llm.invoke(prompt)  # The input is now a string created from the prompt template

    answer = response.content
    st.session_state['flowmessages'].append(AIMessage(content=answer))
    return answer

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
prompt1 = st.text_input("Enter your question here...")

# Button to embed documents
if st.button("Embed Docs"):
    vector_embedding()

# If a question is entered and documents are embedded
if prompt1 and "vectors" in st.session_state:
    # Retrieve last 10 interactions to form the context
    context = get_last_context()
    
    # Create chains for document retrieval and question answering
    document_chain = create_stuff_documents_chain(llm, chat_prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    # Measure the time to get a response
    start = time.process_time()
    
    # Get the chat model response
    answer = get_chatmodel_response(prompt1)
    
    st.write("Response time :", time.process_time() - start)
    
    # Append the interaction to the session state history
    st.session_state.history.append({"question": prompt1, "answer": answer})
    
    # Display the current answer
    st.write(answer)
    
    # With a streamlit expander to show the document similarity search results
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
