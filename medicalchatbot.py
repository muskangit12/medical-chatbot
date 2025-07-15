import os
import json
import uuid
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama3-8b-8192"
SESSIONS_FILE = "chat_sessions.json"

st.set_page_config(page_title="MediWise - Medical RAG Chatbot", page_icon="💊")

@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

def set_custom_prompt():
    template = """
    You are a helpful Medical Assistant.

    Always answer based on the given context.
    If the answer is not present in the context, say:
    "I don't have enough information."

    If the user is asking a follow-up question,
    use the chat history to understand their intent.

    Context: {context}
    Question: {question}

    Answer:
    """
    return PromptTemplate(template=template, input_variables=["context", "question"])

def load_sessions():
    if os.path.exists(SESSIONS_FILE):
        with open(SESSIONS_FILE, "r") as file:
            return json.load(file)
    return {}

def save_sessions(sessions):
    with open(SESSIONS_FILE, "w") as file:
        json.dump(sessions, file)

def main():
    st.sidebar.title("💬 Chat Sessions")

    if "sessions" not in st.session_state:
        st.session_state.sessions = load_sessions()

    if "current_session_id" not in st.session_state:
        new_id = str(uuid.uuid4())
        st.session_state.current_session_id = new_id
        st.session_state.sessions[new_id] = []
        save_sessions(st.session_state.sessions)

    if st.sidebar.button("➕ Start New Chat"):
        new_id = str(uuid.uuid4())
        st.session_state.sessions[new_id] = []
        st.session_state.current_session_id = new_id
        save_sessions(st.session_state.sessions)
        st.rerun()

    if st.session_state.sessions:
        for session_id, messages in st.session_state.sessions.items():
            if messages:
                session_title = messages[0]['user'][:30]
                if st.sidebar.button(session_title, key=session_id):
                    st.session_state.current_session_id = session_id
    else:
        st.sidebar.write("No chats yet.")

    if st.sidebar.button("🗑️ Delete Current Chat"):
        del st.session_state.sessions[st.session_state.current_session_id]
        save_sessions(st.session_state.sessions)
        if st.session_state.sessions:
            st.session_state.current_session_id = list(st.session_state.sessions.keys())[0]
        else:
            new_id = str(uuid.uuid4())
            st.session_state.current_session_id = new_id
            st.session_state.sessions[new_id] = []
        save_sessions(st.session_state.sessions)
        st.rerun()

    st.title("💊 MediWise ")

    current_chat = st.session_state.sessions[st.session_state.current_session_id]

    for pair in current_chat:
        st.chat_message("user").markdown(pair["user"])
        st.chat_message("assistant").markdown(pair["assistant"])

    user_input = st.chat_input("Ask your medical question...")

    if user_input:
        st.chat_message("user").markdown(user_input)
        current_chat.append({"user": user_input, "assistant": ""})
        save_sessions(st.session_state.sessions)

        with st.spinner("Thinking..."):
            try:
                vectorstore = get_vectorstore()
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

                llm = ChatGroq(
                    model_name=GROQ_MODEL_NAME,
                    temperature=0.0,
                    groq_api_key=os.environ["GROQ_API_KEY"]
                )

                qa_chain = ConversationalRetrievalChain.from_llm(
                    llm=llm,
                    retriever=retriever,
                    return_source_documents=False,
                    combine_docs_chain_kwargs={"prompt": set_custom_prompt()}
                )

                response = qa_chain.invoke({
                    "question": user_input,
                    "chat_history": [(m["user"], m["assistant"]) for m in current_chat[:-1]]
                })

                answer = response["answer"]
                st.chat_message("assistant").markdown(answer)

                current_chat[-1]["assistant"] = answer
                save_sessions(st.session_state.sessions)

            except Exception as e:
                st.error(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()
