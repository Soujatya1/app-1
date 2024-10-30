
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

if not os.path.exists("uploaded_files"):
    os.makedirs("uploaded_files")


if 'history' not in st.session_state:
    st.session_state.history = []

if 'last_context' not in st.session_state:
    st.session_state.last_context = ""

uploaded_files = st.file_uploader("Upload a file", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Save each file temporarily in the created directory
        file_path = os.path.join("uploaded_files", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"File '{uploaded_file.name}' uploaded successfully!")

    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("uploaded_files")
        st.session_state.docs = st.session_state.loader.load()
        st.write(f"Loaded {len(st.session_state.docs)} documents.")

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

llm = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")

def create_prompt(input_text):
    previous_interactions = "\n".join(
        [f"You: {h['question']}\nBot: {h['answer']}" for h in st.session_state.history[-5:]]
    )
    return ChatPromptTemplate.from_template(
        f"""
        Answer the questions based on the provided context only.
        Please provide the most accurate response based on the question.
        Previous Context: {st.session_state.last_context}
        Previous Interactions:\n{previous_interactions}
        <context>
        {{context}}
        <context>
        Questions: {input_text}
        """
    )

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

input_box = st.empty()
with input_box.container():
    prompt1 = st.text_input("Enter your question here.....", key="user_input", placeholder="Type your question...")

if prompt1 and "vectors" in st.session_state:
    # Detect the language of the input
    detected_language = detect_language(prompt1)

    # Use detected language only if "Auto-detect" is selected or a blank is chosen
    if selected_language == "Auto-detect":
        if detected_language:
            st.write(f"Detected language: {detected_language}")
        else:
            detected_language = "en"  # Default to English if detection fails
        source_language = detected_language
    else:
        # Use the language from the selectbox
        source_language = language_mapping[selected_language]

    # Translate input to English if necessary
    if source_language != "en":
        translated_prompt = translate_text(prompt1, source_language, "en")
    else:
        translated_prompt = prompt1

    # Continue with document retrieval and processing
    document_chain = create_stuff_documents_chain(llm, create_prompt(translated_prompt))
    retriever = st.session_state.vectors.as_retriever(search_type="similarity", k=2)

    # Retrieve filtered documents (priority given to 'text')
    #filtered_documents = retrieve_documents_with_filter(retriever, translated_prompt, "table")
    #filtered_documents += retrieve_documents_with_filter(retriever, translated_prompt, "text")
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    #filtered_documents_dict = {'documents': filtered_documents}

    #input_dict = {"input": translated_prompt, "documents": filtered_documents_dict['documents']}

    # Timing the response
    start = time.process_time()
    response = retrieval_chain.invoke({'input': translated_prompt})
    st.write("Response time:", time.process_time() - start)

    # Get the answer
    answer = response['answer']

    #answer_with_source = append_source_to_answer(answer, filtered_documents)


    # Translate the answer back to the selected language if needed
    if selected_language != "English" and selected_language != "Auto-detect":
        translated_answer = translate_text(answer, "en", language_mapping[selected_language])
        answer = translated_answer if translated_answer else answer

    if detected_language != "en":
        translated_answer = translate_text(answer, "en", detected_language)
        answer = translated_answer if translated_answer else answer

    # Store the response in session history
    st.session_state.last_context = answer
    st.session_state.history.append({"question": prompt1, "answer": answer})
    st.write(f"**Bot:** {answer}")

    with st.expander("Document Similarity Search"):
        if "context" in response:
            for i, doc in enumerate(response["context"]):
                doc_name = doc.metadata.get("source", "Unknown Document")
                st.write(f"Document: {doc_name}")
                st.write(doc.page_content)
