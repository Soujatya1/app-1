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
import requests  # Make sure to import requests for API calls

st.title("Knowledge Management Chatbot")

if not os.path.exists("uploaded_files"):
    os.makedirs("uploaded_files")

# Initialize the interaction history if not present
if 'history' not in st.session_state:
    st.session_state.history = []

# Keep a context of the last question and answer for dynamic prompts
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

    # Automatically perform the embedding after file upload
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("uploaded_files")
        st.session_state.docs = st.session_state.loader.load()
        st.write(f"Loaded {len(st.session_state.docs)} documents.")

        # Debugging: Print document metadata to check if source names are available
        #for doc in st.session_state.docs:
        #    st.write(f"Document metadata: {doc.metadata}")  # This should show the filename

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Initialize LLM model
llm = ChatGroq(groq_api_key="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", model_name="Llama3-8b-8192")

# Chat Prompt Template with dynamic context
def create_prompt(input_text):
    previous_interactions = "\n".join(
        [f"You: {h['question']}\nBot: {h['answer']}" for h in st.session_state.history[-5:]]  # Last 5 interactions
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

# Add your translation function here
def translate_response(response_text, target_language):
    api_url = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/getModelsPipeline/"
    user_id = "bdeee189dc694351b6b248754a918885"
    ulca_api_key = "099c9c6409-1308-4503-8d33-64cc5e49a07f"
    source_language = "en"

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
            return response_text

    except Exception as e:
        return response_text

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
                    "source": response_text
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

with st.sidebar:
    st.header("Language Selection")
    selected_language = st.selectbox("Select language for translation:", language_options, key="language_selection")

# Dropdown for language selection
selected_language = st.selectbox("Select language for translation:", language_options)

# If a question is entered and documents are embedded
if prompt1 and "vectors" in st.session_state:
    # Create chains for document retrieval and question answering
    document_chain = create_stuff_documents_chain(llm, create_prompt(prompt1))
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measure the time to get a response
    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("Response time :", time.process_time() - start)

    # Extract the answer from the response
    answer = response['answer']

    # Update the last context with the new answer
    st.session_state.last_context = answer

    # Append the interaction to the session state history
    st.session_state.history.append({"question": prompt1, "answer": answer})

    # Translate the answer if it's not in English
    if selected_language != "English":
        translated_answer = translate_response(answer, language_mapping[selected_language])
        answer = translated_answer if translated_answer else answer  # Use the original answer if translation fails

    # Display the current answer
    st.write(answer)

    # With a streamlit expander to show the document similarity search results
    with st.expander("Document Similarity Search"):
        if "context" in response:
            for i, doc in enumerate(response["context"]):
                # Assuming doc contains a 'metadata' field with the filename
                doc_name = doc.metadata.get("source", "Unknown Document")
                st.write(f"Document: {doc_name}")  # Display the document name
                st.write(doc.page_content)
                st.write("--------------------------------")
