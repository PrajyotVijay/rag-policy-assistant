# RAG Policy Assistant (LangChain)

A Retrieval-Augmented Generation (RAG) system that answers employee questions about a company's policies using only the content of a policy document. Built as the hands-on assignment for the IIT Madras workshop **Introduction to LangChain: Chains, Retrieval and RAG**.

If a question isn't covered by the document, the assistant says so instead of inventing an answer.

## How it works

```
Policy doc -> Split into chunks -> Embeddings -> FAISS vector store
                                                      |
Question -> Retriever (top 3 chunks) -> Prompt + LLM -> Answer
```

| Step | Implementation |
|---|---|
| Load | `TextLoader` on the exported Google Doc (TechNova Solutions policy) |
| Split | `RecursiveCharacterTextSplitter` (chunk size 320, overlap 50) |
| Embed | `sentence-transformers/all-MiniLM-L6-v2` via `HuggingFaceEmbeddings` |
| Store | FAISS |
| Retrieve | Top 3 most relevant chunks (`k=3`) |
| Generate | `PromptTemplate` + `ChatGroq` (`openai/gpt-oss-120b`, temperature 0) via an LCEL chain |

## Run it

```bash
pip install -r requirements.txt

# Get a free API key at https://console.groq.com
export GROQ_API_KEY="your-key-here"      # Windows PowerShell: $env:GROQ_API_KEY="your-key-here"

python rag_policy_assistant.py
```

If `GROQ_API_KEY` isn't set, the script prompts for it. The key is never stored in the code.

## Sample output

| Question | Answer |
|---|---|
| How many paid annual leave days does a full-time employee receive per year? | 18 paid annual leave days per calendar year. |
| An employee wants to work remotely on Wednesday and take leave on Friday. What do the policies say? | Combines two chunks: remote work needs an email to the manager one working day ahead plus approval; leave needs a request at least 3 calendar days ahead plus manager confirmation. |
| What is the company's parental leave policy? | "The provided document does not contain this information." |

## Lessons learned

- **Chunk size matters.** My first split (250 characters) cut the annual leave rule mid-sentence. Raising it to 320 kept each policy rule in a single chunk.
- **The prompt is part of the safety net.** Instructing the model to answer only from the context, with an exact fallback sentence, is what makes it refuse on missing information instead of hallucinating.

## Tech stack

Python, LangChain (LCEL), Groq, Hugging Face sentence-transformers, FAISS

## Next steps

- Support multiple PDFs and text files as sources
- Add a Streamlit chat UI
- Show source chunks alongside each answer
- Deploy it
