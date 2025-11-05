# 📚 RAG Demo — LangChain + Gemini + Hugging Face + Chroma

This project demonstrates an **end-to-end Retrieval-Augmented Generation (RAG)** system built using **LangChain**, **Google Gemini (via `langchain_google_genai`)**, **Hugging Face embeddings**, and **ChromaDB**.

It loads a **PDF document**, splits it into chunks, stores them as embeddings in a Chroma vector database, and answers user questions using **Google Gemini 2.5 Flash Lite**, powered by retrieved context from the vector store.

---

## 🚀 Features

- 🧠 Uses **Google Gemini (Generative AI)** for reasoning and question answering.
- 🔍 Uses **Hugging Face embeddings** via the **Hugging Face Inference API**.
- 🗂️ Stores document embeddings in **ChromaDB**, a local vector database.
- 📄 Accepts any **PDF** file as input.
- 🔁 Full **manual RAG pipeline** using LangChain's new **LCEL syntax** (no `RetrievalQA`).
- 💬 Interactive command-line question answering loop.

---

## 🧰 Tech Stack

| Component                | Library                                        |
| ------------------------ | ---------------------------------------------- |
| **Language**             | Python 3.10+                                   |
| **Vector Store**         | [Chroma](https://docs.trychroma.com/)          |
| **Embeddings**           | HuggingFaceEndpointEmbeddings                  |
| **LLM**                  | ChatGoogleGenerativeAI (Gemini 2.5 Flash Lite) |
| **Document Loader**      | LangChain PyPDFLoader                          |
| **Prompting & Chaining** | LangChain Core (LCEL syntax)                   |
| **Env Management**       | python-dotenv                                  |

---

## 📦 Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/rag-demo.git
cd rag-demo
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a file named `.env` in the project root with:

```bash
GOOGLE_API_KEY=your_google_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

⚠️ Do not share this file — your API keys are private. You can provide a safe `.env.example` for reference in GitHub.

You can get free API keys here:

- **Google Gemini** → https://aistudio.google.com/
- **Hugging Face** → https://huggingface.co/settings/tokens

---

## 📄 Usage

### 1️⃣ Add your PDF file

Place your PDF file in a folder called `data/` and update this line in `app.py`:

```python
pdf_path = "data/your_file.pdf"
```

### 2️⃣ Run the Application

```bash
python app.py
```

### 3️⃣ Ask Questions

You'll enter an interactive mode like this:

```
Ask a question (or type 'exit'): What is this document about?
Answer: This PDF describes the motivation and goals of...
```

The app will also display the top retrieved source snippets used to answer.

---

## 🧠 How It Works

**Document Loading**

The PDF is loaded and split into small overlapping chunks.

**Embedding Creation**

Each chunk is embedded into a numerical vector using Hugging Face.

**Vector Storage**

The chunks and embeddings are stored in ChromaDB locally.

**Query Process**

The user's question is embedded and compared against stored vectors. The top-matching chunks are retrieved. Gemini uses these chunks as context to generate a grounded answer.

---

## 🧪 Example Run

```
Ask a question (or type 'exit'): What inspired the author to write this letter?
Answer: The author was motivated by a strong interest in...

Sources (2 documents):
  1. I am writing to express my motivation...
  2. My background in computer science...
```

---

## ⚙️ Future Enhancements

- 🌐 Add a FastAPI or Streamlit frontend
- 🧠 Multi-document retrieval and ranking
- 💾 Cache embeddings and allow re-indexing
- 🧮 Support for model parameter tuning
