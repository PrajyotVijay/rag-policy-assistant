"""
RAG system over the TechNova Solutions policy document using LangChain.
Pipeline: load -> split -> embed -> vector store -> retriever (k=3) -> prompt + LLM.
Requires the GROQ_API_KEY environment variable (not included in this file).
"""
import os
import requests
from getpass import getpass

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# API key from the environment (never hardcoded)
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = getpass("Enter GROQ_API_KEY: ")

# 1. Load the policy document
DOC_ID = "1pBXd2LLKniKW4xa7LRcnUhRiLeSKzAEQEgN6ejLoWHM"
url = f"https://docs.google.com/document/d/{DOC_ID}/export?format=txt"
r = requests.get(url)
r.raise_for_status()
with open("policy.txt", "w", encoding="utf-8") as f:
    f.write(r.text)

loader = TextLoader("policy.txt", encoding="utf-8-sig")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=320, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3. Generate embeddings
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store embeddings in a vector store
vectorstore = FAISS.from_documents(chunks, embedding_model)

# 5. Retriever returning the top 3 chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. Prompt template + LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0,
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """
You are an assistant that answers employee questions about company policies.
Answer ONLY using the context below. If the context does not contain the
information needed, reply exactly: "The provided document does not contain
this information." Do not make up policies.

Context:
{context}

Question: {question}

Answer:
"""
prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. Test questions
questions = [
    "How many paid annual leave days does a full-time employee receive per year?",
    "An employee wants to work remotely on Wednesday and take leave on Friday. What do the company policies say about these two requests?",
    "What is the company's parental leave policy?",
]

if __name__ == "__main__":
    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q}")
        print("Answer:", rag_chain.invoke(q))
        print("=" * 60)

# ---------------------------------------------------------------
# OUTPUTS
# ---------------------------------------------------------------
# Q1: How many paid annual leave days does a full-time employee receive per year?
# Answer: A full-time employee receives **18 paid annual leave days per calendar year**.
#
# Q2: An employee wants to work remotely on Wednesday and take leave on Friday. What do the company policies say about these two requests?
# Answer:
# - Remote work on Wednesday: The policy permits employees to work remotely on Wednesdays (as well as Fridays). The employee must email their manager at least one working day in advance, and the remote-work arrangement is subject to the manager's approval.
#
# - Leave on Friday: Annual leave may be taken on any day, but the employee must submit a leave request to their manager at least 3 calendar days before the requested leave date (Friday). The leave is granted only after the manager confirms it.
#
# Q3: What is the company's parental leave policy?
# Answer: The provided document does not contain this information.
