import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load .env file
load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Constants
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
HUGGINGFACE_MODEL_ID = "google/flan-t5-large"
DB_FAISS_PATH = "vectorstore/db_faiss"

st.set_page_config(page_title="Medical QA Bot", page_icon="💊")

@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

@st.cache_resource
def load_llm():
    return HuggingFaceHub(
        repo_id=HUGGINGFACE_MODEL_ID,
        huggingfacehub_api_token=hf_token,
        model_kwargs={"temperature": 0.5, "max_length": 512}
    )

def get_prompt():
    prompt_template = """
    Answer the question only using the provided context.
    If not in context, say "I don't have enough information."

    Context: {context}
    Question: {question}

    Answer:
    """
    return PromptTemplate(input_variables=["context", "question"], template=prompt_template)

def main():
    st.title("💊 Medical QA Chatbot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for chat in st.session_state.chat_history:
        st.chat_message(chat["role"]).markdown(chat["content"])

    user_query = st.chat_input("Ask your medical question...")

    if user_query:
        st.chat_message("user").markdown(user_query)
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.spinner("💬 Thinking..."):
            try:
                db = load_vectorstore()
                llm = load_llm()
                prompt = get_prompt()

                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=db.as_retriever(search_kwargs={"k": 3}),
                    chain_type="stuff",
                    chain_type_kwargs={"prompt": prompt},
                    return_source_documents=True
                )

                response = qa_chain.invoke({"query": user_query})

                answer = response["result"]
                st.chat_message("assistant").markdown(answer)
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"⚠️ {str(e)}")

if __name__ == "__main__":
    main()
