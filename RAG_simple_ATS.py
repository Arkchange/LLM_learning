from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain_core.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

import os

loader = PyPDFLoader("CV2e_KEVIN_DS.pdf")
document = loader.load()

document[0].page_content[:500]

splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 20)
chunks = splitter.split_documents(document)

chunks

from huggingface_hub import login
login(token="TOKEN")

embeddind_model = HuggingFaceBgeEmbeddings(model_name = "all-MiniLM-L6-v2")

faiss_db = FAISS.from_documents(chunks, embeddind_model)
faiss_db.save_local("faiss_local")

retriever = faiss_db.as_retriever(search_kwargs = {"k" : 2})
retriever.get_relevant_documents("What programming languages does the candidate know?")

query = "What programming languages does the candidate know?"
results = faiss_db.similarity_search(query, k=3)
results

model_id = "microsoft/phi-2" #had to choose a small model

tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto", token=True)

llm = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=150)

docs = faiss_db.similarity_search(query, k=5)
context = "\n\n".join(doc.page_content for doc in docs)

template = """Analyze this CV and answer the question:

CV CONTEXT:
{context}

QUESTION: {question}

Answer with:
1. Suitability (YES/NO)
2. Key qualifications
3. Missing skills (if any)"""

prompt_template = PromptTemplate.from_template(template)
prompt_template

from langchain_core.runnables import RunnableLambda

def extract_prompt_text(prompt_value):
    return prompt_value.text

def extract_llm_output(llm_output):
    return llm_output[0]['generated_text']

query = "The CV of Kevin is fit for a role in the data field?"

chain = (
    prompt_template
    | RunnableLambda(extract_prompt_text)
    | llm
    | RunnableLambda(extract_llm_output)
    | StrOutputParser()
)

try:
    result = chain.invoke({"context": context, "question": query})
    print("Answer:", result)
except Exception as e:
    print(f"Error processing request: {str(e)}")

template_decision = """Given this answer {context}, answer wwith either YES or NO for a role in data as an neutral hiring manager and why

Answer :"""

prompt_template_decision = PromptTemplate.from_template(template_decision)

chain_decision = (
    prompt_template_decision
    | RunnableLambda(extract_prompt_text)
    | llm
    | RunnableLambda(extract_llm_output)
    | StrOutputParser()
)

chain_decision.invoke({"context": result})