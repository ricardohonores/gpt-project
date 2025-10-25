# GPT Project: RAG & Fine-tuning

[Español](README.es.md) | English

Experimentation project with advanced AI techniques: **RAG (Retrieval-Augmented Generation)** and **Fine-tuning** using local language models.

## 🎯 Description

This project implements two complementary approaches to enhance language model capabilities:

1. **RAG System**: Allows a local LLM to answer questions based on your own documents using vector search
2. **Fine-tuning with QLoRA**: Adapts a pre-trained model to specific tasks using 4-bit quantization

## ✨ Features

### RAG System (`chatrag_py.py`)
- 📚 Processes PDF and TXT documents
- 🔍 Semantic search with FAISS
- 🤖 Integration with Ollama for local inference
- 💬 Interactive Q&A interface
- 📝 Shows information sources in each response

### Fine-tuning (`finetuning.py`)
- ⚡ QLoRA (Quantized Low-Rank Adaptation) for training with low VRAM
- 🎯 4-bit quantization
- 💾 Saves only adapters (lightweight weights)
- 🔧 Configurable for GPU or CPU

## 📋 Requirements

- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- `gpt-oss:20b` model downloaded in Ollama
- GPU with CUDA (optional, but recommended for fine-tuning)

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/ricardohonores/gpt-project.git
cd gpt-project
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

**For RAG system:**
```bash
pip install langchain-community langchain-core transformers accelerate sentence-transformers faiss-cpu pypdf
```

**For fine-tuning (additional):**
```bash
pip install datasets peft trl bitsandbytes
```

**For GPU (optional):**
```bash
pip install faiss-gpu
```

### 4. Install and configure Ollama
```bash
# Install Ollama (if you don't have it)
curl -fsSL https://ollama.ai/install.sh | sh

# Download the model
ollama pull gpt-oss:20b
```

## 💻 Usage

### RAG System

1. **Place your documents** in the `mis_documentos/` folder (.txt or .pdf files)

2. **Start Ollama** (in another terminal):
```bash
ollama serve
```

3. **Run the RAG system**:
```bash
python chatrag_py.py
```

4. **Ask questions** interactively. Type `salir` to exit.

**Example:**
```
Your question: What is generative AI?

--- Answer ---
Generative AI is artificial intelligence that creates new content...

--- Sources ---
-> Source: /path/to/document.pdf, Page: 5
```

### Fine-tuning

1. **Prepare your dataset** in JSONL format (`dataset.jsonl`):
```json
{"instruction": "Question or task", "output": "Expected response"}
{"instruction": "Another question", "output": "Another response"}
```

2. **Run training**:
```bash
python finetuning.py
```

3. **Adapters are saved** in `./gpt_oss_fine_tuned/`

## 📁 Project Structure

```
gpt-project/
├── chatrag_py.py          # Complete RAG system
├── finetuning.py          # Fine-tuning script with QLoRA
├── dataset.jsonl          # Sample dataset
├── CLAUDE.md              # Technical documentation (English)
├── CLAUDE.es.md           # Technical documentation (Spanish)
├── README.md              # This file (English)
├── README.es.md           # README in Spanish
├── .gitignore             # Files ignored by git
└── mis_documentos/        # Document folder for RAG
    ├── *.pdf
    └── *.txt
```

## 🔧 Configuration

### RAG System
Edit the following variables in `chatrag_py.py`:
```python
DOCS_FOLDER = "/path/to/your/documents"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "gpt-oss:20b"
```

### Fine-tuning
Edit the following variables in `finetuning.py`:
```python
OLLAMA_MODEL_NAME = "gpt-oss:20b"
DATASET_FILE = "dataset.jsonl"
OUTPUT_DIR = "./gpt_oss_fine_tuned"
```

## 🤔 RAG or Fine-tuning?

| Feature | RAG | Fine-tuning |
|---------|-----|-------------|
| **Modifies the model** | ❌ No | ✅ Yes |
| **Requires training** | ❌ No | ✅ Yes |
| **Dynamic documents** | ✅ Yes | ❌ No |
| **Shows sources** | ✅ Yes | ❌ No |
| **Changes behavior** | ❌ No | ✅ Yes |
| **Memory usage** | 🟢 Low | 🟡 Medium |

**Use RAG when:** You need to answer questions about specific documents or knowledge that changes frequently.

**Use Fine-tuning when:** You want the model to learn a specific style, technical domain, or new behavior patterns.

## 📚 Additional Documentation

For technical details about architecture and configuration, see [CLAUDE.md](CLAUDE.md).

## 📄 License

This project is open source and available for educational and experimental use.

## 🤝 Contributions

Contributions are welcome. Feel free to open issues or pull requests.
