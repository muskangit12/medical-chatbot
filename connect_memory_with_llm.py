import os
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in environment variables")

def load_llm():
    return HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        huggingfacehub_api_token=HF_TOKEN,
        temperature=0.5,
        max_length=512
    )

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

def set_custom_prompt():
    return PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )

# Load Database
DB_FAISS_PATH = "vectorstore/db_faiss"
try:
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
except Exception as e:
    print(f"Error loading database: {e}")
    exit(1)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=load_llm(),
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': set_custom_prompt()}
)

# Query loop
while True:
    try:
        user_query = input("\nWrite Query Here (or 'quit' to exit): ")
        if user_query.lower() == 'quit':
            break
            
        response = qa_chain.invoke({'query': user_query})
        
        print("\nRESULT:", response["result"])
        print("\nSOURCES:")
        for i, doc in enumerate(response["source_documents"], 1):
            print(f"{i}. {doc.metadata.get('source', 'Unknown source')}")
            print(f"   Excerpt: {doc.page_content[:100]}...\n")
            
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error processing query: {e}")