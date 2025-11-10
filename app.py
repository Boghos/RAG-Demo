import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def load_environment():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable")
    
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise ValueError("Please set HUGGINGFACE_API_KEY environment variable")


def index_pdf(pdf_path: str, persist_dir: str = "chroma_store"):
    # 1 load PDF
    loader = PyPDFLoader(pdf_path)
    docs = loader.load_and_split()
    
    # 2 split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=10
    )
    chunked_docs = splitter.split_documents(docs)
    
    # 3 create embeddings using Hugging Face Inference API
    embed_model = HuggingFaceEndpointEmbeddings(
        huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY")
    )
    
    # 4 create or load chroma vector store
    vectordb = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embed_model,
        persist_directory=persist_dir,
        collection_name="pdf_rag_collection"
    )
    
    return vectordb


def create_qa_chain(vectordb):
    # create retriever
    retriever = vectordb.as_retriever(search_kwargs={'k': 20})
    
    # create LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
    
    # define prompt
    prompt_template = PromptTemplate.from_template(
        """You are a helpful assistant analyzing a document. Answer the question based on the context provided.
    If the question asks for a name or author, look for it at the end of the document (often after "Sincerely" or "Best regards").
    
    Context: {context}
    
    Question: {question}
    
    Answer:"""
    )
    
    # create the chain using LCEL
    qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    return qa_chain, retriever


def ask_question(qa_chain, retriever, question: str):
    answer = qa_chain.invoke(question)
    source_docs = retriever.invoke(question)
    return {"answer": answer, "sources": source_docs}


def main():
    load_environment()
    # Update this with your actual PDF file path
    pdf_path = "data/Boghos_Hamalian_Motivation_Letter.pdf"
    persist_dir = "chroma_store"
    
    # Verify file exists
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file '{pdf_path}' not found")
        print(f"Current directory: {os.getcwd()}")
        return
    
    # Index if not done yet
    vectordb = index_pdf(pdf_path, persist_dir=persist_dir)
    
    # Create QA chain
    qa_chain, retriever = create_qa_chain(vectordb)
    
    # Interactive loop
    while True:
        q = input("\nAsk a question (or type 'exit'): ")
        if q.strip().lower() == "exit":
            break
        result = ask_question(qa_chain, retriever, q)
        print("Answer:", result['answer'])
        if result.get('sources'):
            print(f"\nSources ({len(result['sources'])} documents):")
            for i, doc in enumerate(result['sources'][:2], 1):
                print(f"  {i}. {doc.page_content[:100]}...")


if __name__ == "__main__":
    main()