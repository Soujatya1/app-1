
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
import requests
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

st.title("Document Comparer!")
st.subheader("Compare Your Documents")

DetectorFactory.seed = 0

def detect_language(text):
    try:
        return detect(text)
    except LangDetectException:
        return None

st.markdown("""
    <style>
    .input-box {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        padding: 10px;
        box-shadow: 0 -1px 5px rgba(0, 0, 0, 0.1);
        z-index: 999;
    }
    .conversation-history {
        max-height: 75vh;
        overflow-y: auto;
    }
    </style>
    """, unsafe_allow_html=True)

# Ensure directories for both upload slots
os.makedirs("uploaded_files_1", exist_ok=True)
os.makedirs("uploaded_files_2", exist_ok=True)

# Session state for history and context
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_context' not in st.session_state:
    st.session_state.last_context = ""
    
llm = ChatGroq(groq_api_key="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", model_name="Llama3-70b-8192")
# First uploader for Document Set 1
uploaded_files_1 = st.file_uploader("Upload files for Document Set 1", type=["pdf"], accept_multiple_files=True, key="uploader_1")
if uploaded_files_1:
    for uploaded_file in uploaded_files_1:
        file_path = os.path.join("uploaded_files_1", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded to Document Set 1.")

    if "vectors_1" not in st.session_state:
        st.session_state.embeddings_1 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader_1 = PyPDFDirectoryLoader("uploaded_files_1")
        st.session_state.docs_1 = st.session_state.loader_1.load()
        st.write(f"Loaded {len(st.session_state.docs_1)} documents for Set 1.")

        st.session_state.text_splitter_1 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        st.session_state.final_documents_1 = st.session_state.text_splitter_1.split_documents(st.session_state.docs_1)
        st.session_state.vectors_1 = FAISS.from_documents(st.session_state.final_documents_1, st.session_state.embeddings_1)

# Second uploader for Document Set 2
uploaded_files_2 = st.file_uploader("Upload files for Document Set 2", type=["pdf"], accept_multiple_files=True, key="uploader_2")
if uploaded_files_2:
    for uploaded_file in uploaded_files_2:
        file_path = os.path.join("uploaded_files_2", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File '{uploaded_file.name}' uploaded to Document Set 2.")

    if "vectors_2" not in st.session_state:
        st.session_state.embeddings_2 = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader_2 = PyPDFDirectoryLoader("uploaded_files_2")
        st.session_state.docs_2 = st.session_state.loader_2.load()
        st.write(f"Loaded {len(st.session_state.docs_2)} documents for Set 2.")

        st.session_state.text_splitter_2 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        st.session_state.final_documents_2 = st.session_state.text_splitter_2.split_documents(st.session_state.docs_2)
        st.session_state.vectors_2 = FAISS.from_documents(st.session_state.final_documents_2, st.session_state.embeddings_2)

# Language translation setup (reuse existing code for translation)

# Chat and retrieval function
def create_prompt(input_text):
    previous_interactions = "\n".join(
        [f"You: {h['question']}\nBot: {h['answer']}" for h in st.session_state.history[-5:]]
    )
    return ChatPromptTemplate.from_template(
        f"""
        Answer the questions based on the provided context only.
        Previous Context: {st.session_state.last_context}
        Previous Interactions:\n{previous_interactions}
        <context>
        {{context}}
        <context>
        Question: {input_text}
        """
    )

# Language selection (reuse existing code for language mapping and detection)
def translate_text(text, source_language, target_language):
    api_url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline/"
    user_id = "bdeee189dc694351b6b248754a918885"
    ulca_api_key = "099c9c6409-1308-4503-8d33-64cc5e49a07f"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ulca_api_key}",
        "userID": user_id,
        "ulcaApiKey": ulca_api_key
    }

    payload = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": source_language,
                        "targetLanguage": target_language
                    }
                }
            }
        ],
        "pipelineRequestConfig": {
            "pipelineId": "64392f96daac500b55c543cd"
        }
    }

    try:
        response = requests.post(api_url, json=payload, headers=headers)
        if response.status_code == 200:
            response_data = response.json()
            service_id = response_data["pipelineResponseConfig"][0]["config"][0]["serviceId"]
        else:
            return text

    except Exception as e:
        return text

    compute_payload = {
        "pipelineTasks": [
            {
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": source_language,
                        "targetLanguage": target_language
                    },
                    "serviceId": service_id
                }
            }
        ],
        "inputData": {
            "input": [
                {
                    "source": text
                }
            ]
        }
    }

    callback_url = response_data["pipelineInferenceAPIEndPoint"]["callbackUrl"]
    headers2 = {
        "Content-Type": "application/json",
        response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["name"]:
            response_data["pipelineInferenceAPIEndPoint"]["inferenceApiKey"]["value"]
    }

    try:
        compute_response = requests.post(callback_url, json=compute_payload, headers=headers2)
        if compute_response.status_code == 200:
            compute_response_data = compute_response.json()
            translated_content = compute_response_data["pipelineResponse"][0]["output"][0]["target"]
            return translated_content
        else:
            return ""

    except Exception as e:
        return ""

language_mapping = {
    "Auto-detect": "",
    "English": "en",
    "Kashmiri": "ks",
    "Nepali": "ne",
    "Bengali": "bn",
    "Marathi": "mr",
    "Sindhi": "sd",
    "Telugu": "te",
    "Gujarati": "gu",
    "Gom": "gom",
    "Urdu": "ur",
    "Santali": "sat",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Manipuri": "mni",
    "Tamil": "ta",
    "Hindi": "hi",
    "Punjabi": "pa",
    "Odia": "or",
    "Dogri": "doi",
    "Assamese": "as",
    "Sanskrit": "sa",
    "Bodo": "brx",
    "Maithili": "mai"
}

language_options = list(language_mapping.keys())

st.header("Conversation History")
for interaction in st.session_state.history:
    st.write(f"**You:** {interaction['question']}")
    st.write(f"**Bot:** {interaction['answer']}")
    st.write("---")

st.write("---")

with st.sidebar:
    st.header("Language Selection")
    selected_language = st.selectbox("Select language for translation:", language_options, key="language_selection")

# Input box for user query
input_box = st.empty()
with input_box.container():
    prompt1 = st.text_input("Enter your question here...", key="user_input", placeholder="Type your question...")

if prompt1 and "vectors_1" in st.session_state and "vectors_2" in st.session_state:
    # Detect the language of the input
    detected_language = detect_language(prompt1)
    source_language = detected_language or "en"

    # Translate input to English if necessary
    translated_prompt = prompt1  # Modify this if translation is required

    # Document Set 1 retrieval
    document_chain_1 = create_stuff_documents_chain(llm, create_prompt(translated_prompt))
    retriever_1 = st.session_state.vectors_1.as_retriever(search_type="similarity", k=2)
    retrieval_chain_1 = create_retrieval_chain(retriever_1, document_chain_1)
    response_1 = retrieval_chain_1.invoke({'input': translated_prompt})
    answer_1 = response_1['answer']

    # Document Set 2 retrieval
    document_chain_2 = create_stuff_documents_chain(llm, create_prompt(translated_prompt))
    retriever_2 = st.session_state.vectors_2.as_retriever(search_type="similarity", k=2)
    retrieval_chain_2 = create_retrieval_chain(retriever_2, document_chain_2)
    response_2 = retrieval_chain_2.invoke({'input': translated_prompt})
    answer_2 = response_2['answer']

    # Display answers
    st.write("**Document Set 1 Answer:**", answer_1)
    st.write("**Document Set 2 Answer:**", answer_2)

    # Save responses to session history
    st.session_state.history.append({"question": prompt1, "answer_set_1": answer_1, "answer_set_2": answer_2})

    # Optionally display retrieved document contents
    with st.expander("Document Set 1 Similarity Search"):
        for i, doc in enumerate(response_1.get("context", [])):
            doc_name = doc.metadata.get("source", "Unknown Document")
            st.write(f"Document: {doc_name}")
            st.write(doc.page_content)

    with st.expander("Document Set 2 Similarity Search"):
        for i, doc in enumerate(response_2.get("context", [])):
            doc_name = doc.metadata.get("source", "Unknown Document")
            st.write(f"Document: {doc_name}")
            st.write(doc.page_content)
