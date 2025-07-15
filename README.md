# 💊 MediWise - Medical RAG Chatbot

MediWise is a Retrieval-Augmented Generation (RAG) based Medical Chatbot designed to assist users with medical-related questions using context-based answers.

Built with **LangChain**, **FAISS**, and **Streamlit**, this chatbot retrieves information from a pre-built vector store and interacts through a user-friendly interface.

---

## 🚀 Features
- 📝 Context-aware Q&A based on medical knowledge
- 🗂️ Multi-session chat history with option to delete chats
- 💾 Chat history stored and persisted between sessions
- 🧩 Uses HuggingFace Embeddings and Groq Llama-3 Model
- 🖥️ Built with Streamlit for web deployment

---

## 🛠️ Tech Stack
- `Python`
- `Streamlit`
- `LangChain`
- `FAISS`
- `HuggingFace Sentence Transformers`
- `LangChain-Groq`
- `Streamlit Cloud Deployment`

---

## ⚙️ How to Run Locally

1️⃣ **Clone the Repository**
```bash
git clone https://github.com/muskangit12/medical-chatbot.git
cd medical-chatbot
```

2️⃣ **Install Dependencies**
```bash
pip install -r requirements.txt
```

3️⃣ **Set your API Key in `.env`**
```
GROQ_API_KEY=your_groq_api_key_here
```

4️⃣ **Run the App**
```bash
streamlit run medicalchatbot.py
```

---

## 🌐 Deployment
Deployed on **Streamlit Cloud**
➡️ Visit: [https://mediwise.streamlit.app/]

---
