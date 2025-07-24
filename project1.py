import streamlit as st
import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from tempfile import NamedTemporaryFile

# Set API Key from .streamlit/secrets.toml
#os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["OPENROUTER_API_KEY"] = st.secrets["OPENROUTER_API_KEY"]

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("📄 Chat with your PDF")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        # Save uploaded PDF to a temp file
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load and split documents
        loader = PyPDFLoader(tmp_path)
        pages = loader.load_and_split()

        # Create vector store
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(pages, embedding=embeddings)

        # Create conversation chain
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(temperature=0),
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        # Initialize session state
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("Ask a question about your PDF:")

        if query:
            with st.spinner("Getting answer..."):
                result = qa_chain.run(query)
                st.session_state.chat_history.append((query, result))

        # Show history
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for q, a in reversed(st.session_state.chat_history):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Bot:** {a}")