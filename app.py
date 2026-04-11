import streamlit as st
import os
from PyPDF2 import PdfReader
import faiss
import numpy as np
from openai import OpenAI

# -----------------------------
# Setup
# -----------------------------
st.set_page_config(page_title="Gov AI Assistant", layout="wide")

# ✅ Temporary API key fallback (REMOVE before GitHub push)
#API_KEY = os.getenv("OPENAI_API_KEY") or "sk-xxxxxxxxxxxx"
API_KEY = "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"
client = OpenAI(api_key=API_KEY)

if not API_KEY:
    st.error("⚠️ Please set OPENAI_API_KEY environment variable")
    st.stop()

client = OpenAI(api_key=API_KEY)

st.title("🏛️ Govt AI Assistant (MVP)")
st.write("AI-powered assistant for Governance, Audit, Health & Panchayat use-cases")

# -----------------------------
# Helper Functions
# -----------------------------

def ask_llm(prompt, system_prompt="You are a helpful government assistant."):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content


@st.cache_data(show_spinner=False)
def get_pdf_text(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


@st.cache_resource(show_spinner=False)
def build_vector_store(text):
    chunks = chunk_text(text)
    embeddings = []

    for chunk in chunks:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=chunk
        )
        embeddings.append(response.data[0].embedding)

    embeddings = np.array(embeddings)
    dim = len(embeddings[0])

    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    return index, chunks


def search_index(query, index, chunks):
    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    ).data[0].embedding

    D, I = index.search(np.array([query_embedding]), k=3)
    results = [chunks[i] for i in I[0]]
    return "\n".join(results)


# -----------------------------
# Sidebar Navigation
# -----------------------------
option = st.sidebar.selectbox(
    "Select Module",
    [
        "📄 Document Q&A",
        "📝 Memo Generator",
        "🤖 General Chatbot",
        "💰 Audit Compliance Checker",
        "🏥 Health Citizen Chatbot"
    ]
)

# -----------------------------
# Module 1: Document Q&A
# -----------------------------
if option == "📄 Document Q&A":
    st.header("📄 Ask Questions from Policy Document")

    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    if uploaded_file:
        text = get_pdf_text(uploaded_file)

        with st.spinner("Processing document..."):
            index, chunks = build_vector_store(text)

        query = st.text_input("Ask your question")

        if query:
            context = search_index(query, index, chunks)

            prompt = f"""
Answer ONLY from the below document.

Context:
{context}

Question:
{query}
"""
            answer = ask_llm(prompt)
            st.success(answer)


# -----------------------------
# Module 2: Memo Generator
# -----------------------------
elif option == "📝 Memo Generator":
    st.header("📝 Government Memo Generator")

    issue = st.text_area("Enter Issue")

    if st.button("Generate Memo"):
        prompt = f"""
Draft an official memo for:
{issue}

Format:
- Subject
- Description
- Action Required
- Signature
"""
        result = ask_llm(prompt)
        st.success(result)


# -----------------------------
# Module 3: General Chatbot
# -----------------------------
elif option == "🤖 General Chatbot":
    st.header("🤖 Citizen Helpdesk")

    user_query = st.text_input("Ask your question")

    if user_query:
        result = ask_llm(user_query)
        st.success(result)


# -----------------------------
# Module 4: Audit CSV
# -----------------------------
elif option == "💰 Audit Compliance Checker":
    st.header("💰 Audit Compliance Checker")

    uploaded_csv = st.file_uploader("Upload Expense File", type=["csv"])

    if uploaded_csv:
        import pandas as pd
        df = pd.read_csv(uploaded_csv)

        st.dataframe(df.head())

        if st.button("Check Compliance"):
            prompt = f"""
Analyze this expense data:

{df.head(30).to_string()}

Find violations, risks, anomalies.
"""
            result = ask_llm(prompt)
            st.success(result)


# -----------------------------
# Module 5: Health Chatbot (CASE STUDY 16)
# -----------------------------
elif option == "🏥 Health Citizen Chatbot":
    st.header("🏥 Health Department Chatbot")

    st.write("Ask about vaccination, hospitals, public health.")

    user_query = st.text_input("Enter your health query")

    if user_query:
        result = ask_llm(
            user_query,
            system_prompt="You are a government health officer. Answer based on Indian public health guidelines."
        )
        st.success(result)