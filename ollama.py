import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama
from tempfile import NamedTemporaryFile

st.set_page_config(page_title="PDF Chatbot (Local)", layout="wide")
st.title("💬 Chat with your PDF — Fully Offline!")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Indexing PDF..."):
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(pages, embedding=embeddings)

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        llm = Ollama(model="mistral")  # change to llama3 or gemma if you want

        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        query = st.text_input("Ask something about your PDF:")

        if query:
            with st.spinner("Thinking..."):
                answer = qa_chain.run(query)
                st.session_state.chat_history.append((query, answer))

        if st.session_state.chat_history:
            st.subheader("Chat History")
            for q, a in reversed(st.session_state.chat_history):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Bot:** {a}")