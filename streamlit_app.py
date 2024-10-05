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

# Initialize session state to store chat history if not already initialized
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Custom CSS to fix the input box at the bottom and style chat messages
st.markdown("""
    <style>
    .input-box {
        position: fixed;
        bottom: 0;
        width: 100%;
        padding: 10px;
        background-color: #ffffff;
        border-top: 1px solid #ddd;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
    }
    .chat-box {
        max-height: 500px;
        overflow-y: auto;
    }
    .user-message {
        background-color: #e0f7fa;
        border-left: 5px solid #00796b;
        padding: 10px;
        margin: 5px 0;
    }
    .bot-message {
        background-color: #ffe0b2;
        border-left: 5px solid #f57c00;
        padding: 10px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Upload file
uploaded_files = st.file_uploader("Upload a file", type=["pdf"], accept_multiple_files=True)

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

    - Think step by step before providing a detailed answer.
    - Answer in a point-wise format when requested.
    - If the user asks for tabular format, try to present information in a table-like structure.
    - Always refer to the conversation history when applicable.
    Example interaction:
    User: What is the summary of the first document?
    Bot: Provides the summary.
    User: Can you provide this in point-wise format?
    Bot: Reformats the previous response into a point-wise list.
    User: Can you present it in a table format?
    Bot: Reformats the same information into a table-like structure.

    <context>
    {context}
    </context>

    Question: {input}
    """)

    # Stuff Document Chain Creation
    document_chain = create_stuff_documents_chain(llm, prompt)

    # Retriever from vector store
    retriever = vector_db.as_retriever()

    # Create a retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Display the chat history
    st.write("### Chat History")
    chat_container = st.container()

    with chat_container:
        for chat in st.session_state['chat_history']:
            st.markdown(f"<div class='user-message'><strong>User:</strong> {chat['user']}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='bot-message'><strong>Bot:</strong> {chat['bot']}</div>", unsafe_allow_html=True)

    # Chat interface
    user_question = st.text_input("Ask a question about the relevant document", key=f"input_{len(st.session_state['chat_history'])}", placeholder="Type your question here...")

    if user_question:
        # Get response from the retrieval chain with context
        response = retrieval_chain.invoke({
            "input": user_question
        })
        
        if response:
            bot_answer = response['answer']
            
            # Update chat history
            st.session_state['chat_history'].append({
                'user': user_question,
                'bot': bot_answer
            })
            
            # Display the updated chat
            chat_container.empty()  # Clear the chat display before updating
            for chat in st.session_state['chat_history']:
                st.markdown(f"<div class='user-message'><strong>User:</strong> {chat['user']}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='bot-message'><strong>Bot:</strong> {chat['bot']}</div>", unsafe_allow_html=True)

        # Display a new input box after the latest message
        user_question = st.text_input("Ask a question about the relevant document", key=f"input_{len(st.session_state['chat_history'])}", placeholder="Type your question here...")
