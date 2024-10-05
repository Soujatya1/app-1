import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings

# App Title
st.title("Knowledge Management Chatbot")

# Custom CSS to fix the input box at the bottom
st.markdown("""
    <style>
    .chat-container {
        display: flex;
        flex-direction: column;
        justify-content: flex-end;
        height: 80vh;
        border: 1px solid #ddd;
        padding: 20px;
        border-radius: 10px;
    }
    .chat-history {
        flex-grow: 1;
        overflow-y: auto;
        margin-bottom: 20px;
    }
    .input-box {
        position: fixed;
        bottom: 0;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state to store chat history
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Upload multiple files
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_documents = []

    # Iterate over each uploaded file
    for uploaded_file in uploaded_files:
        # Save each file temporarily
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Success message
        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

        # Load the PDF
        loader = PyPDFLoader(uploaded_file.name)
        docs = loader.load()

        # Text Splitting into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=15)
        documents = text_splitter.split_documents(docs)

        # Append the documents from the current file to the list
        all_documents.extend(documents)

    # Initialize embeddings and LLM
    hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGroq(groq_api_key="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", model_name='llama3-70b-8192', temperature=0, top_p=0.2)

    # Vector database storage for all documents
    vector_db = FAISS.from_documents(all_documents, hf_embedding)

    # Craft ChatPrompt Template
    prompt = ChatPromptTemplate.from_template("""
    You are a Knowledge Management specialist. Also, wherever possible understand and return the source name of the document from where the information has been pulled.
    Answer the following questions based only on the provided context, previous responses, and the uploaded documents.
    Think step by step before providing a detailed answer.
    Wherever required, answer in a point-wise format.
    Do not answer any unrelated questions which are not in the provided documents, please be careful on this.
    I will tip you with a $1000 if the answer provided is helpful.
    
    <context>
    {context}
    </context>
    Conversation History:
    {chat_history}

    Question: {input}
    """)

    # Stuff Document Chain Creation
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retriever from vector store
    retriever = vector_db.as_retriever()

    # Create a retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Chat interface container with flex layout
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

    # Display chat history in a scrollable area
    st.markdown("<div class='chat-history'>", unsafe_allow_html=True)
    if st.session_state['chat_history']:
        for chat in st.session_state['chat_history']:
            st.markdown(f"<div style='padding: 10px; border-radius: 10px; background-color: #DCF8C6;'><strong>You:</strong> {chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='padding: 10px; border-radius: 10px; background-color: #ECECEC; margin-top: 5px;'><strong>Bot:</strong> {chat['bot']}</div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Placeholder for user input at the bottom of the screen
    st.markdown("<div class='input-box'>", unsafe_allow_html=True)
    user_question = st.text_input("Ask a question about the relevant document", key="input")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # Closing chat container div

    if user_question:
        # Build conversation history
        conversation_history = ""
        for chat in st.session_state['chat_history']:
            conversation_history += f"You: {chat['user']}\nBot: {chat['bot']}\n"

        # Get response from the retrieval chain with context
        response = retrieval_chain.invoke({
            "input": user_question,
            "chat_history": conversation_history
        })

        # Add the user's question and the model's response to chat history
        st.session_state.chat_history.append({"user": user_question, "bot": response['answer']})
