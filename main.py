from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from app import index_pdf, create_qa_chain

load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    vectordb = index_pdf("data/Boghos_Hamalian_Motivation_Letter.pdf", persist_dir="chroma_store")
    app.state.qa_chain, app.state.retriever = create_qa_chain(vectordb)
    print("RAG system initialized")
    yield
    print("Shutting down")

app = FastAPI(title="RAG Chatbot API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]

@app.get("/")
async def root():
    return {"message": "RAG Chatbot API is running"}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    answer = app.state.qa_chain.invoke(request.question)
    source_docs = app.state.retriever.invoke(request.question)
    sources = [doc.page_content[:200] for doc in source_docs[:2]]
    
    return QueryResponse(answer=answer, sources=sources)