import streamlit as st
import pandas as pd
import PyPDF2
from langchain.llm import LLM
from langchain.vectorstores import FAISS
from langchain.chains import ChatChain
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load the pre-trained Langchain model and tokenizer
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Create a Langchain vector store
vector_store = FAISS(model, tokenizer)

# Create a Faiss index
index = vector_store.index

# Create a ChatGroq LLM
llm = LLM("chatgroq-llm")

# Create a ChatChain
chain = ChatChain(llm, vector_store)

# Create a Streamlit application
st.title("Document Comparer Bot")

# Get user input
uploaded_files = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)

# Process the uploaded files
if uploaded_files:
    documents = []
    for file in uploaded_files:
        with file as f:
            pdf_reader = PyPDF2.PdfFileReader(f)
            num_pages = pdf_reader.numPages
            text = ''
            for page in range(num_pages):
                page_obj = pdf_reader.getPage(page)
                text += page_obj.extractText()
            documents.append(text)

    # Create a Pandas dataframe
    df = pd.DataFrame({"text": documents})

    # Add the preprocessed documents to the index
    vector_store.add(df["text"])

    # Get user input
    user_input = st.text_area("Enter a document or a piece of text")

    # Use the ChatChain to generate a response
    response = chain.generate(user_input)

    # Use Faiss to search for similar documents
    distances, indices = index.search(vector_store.encode(user_input), k=5)

    # Calculate the similarity scores
    similarity_scores = [(df.iloc[idx]["text"], 1 - distances[0][i]) for i, idx in enumerate(indices[0])]

    # Display the results
    st.write("Similar documents:")
    for i, (doc, score) in enumerate(similarity_scores):
        st.write(f"{i+1}. {doc} (Similarity score: {score:.2f})")

    # Display the differences between the documents
    st.write("Differences between the documents:")
    for i, (doc, score) in enumerate(similarity_scores):
        differences = []
        for sentence in doc.split("."):
            if sentence not in user_input:
                differences.append(sentence)
        st.write(f"{i+1}. {', '.join(differences)}")
