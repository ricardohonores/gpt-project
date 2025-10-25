# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based GPT experimentation project that implements two complementary AI techniques:
1. **RAG (Retrieval-Augmented Generation)**: Query documents using vector search to provide context to a local LLM
2. **Fine-tuning**: Adapt a pre-trained model using QLoRA (Quantized Low-Rank Adaptation)

The project uses Ollama to run the `gpt-oss:20b` model locally and is designed to work with both GPU (CUDA) and CPU.

## Architecture

### RAG System (`chatrag_py.py`)
- **LLM Integration**: Connects to Ollama API to use the `gpt-oss:20b` model
- **Vector Store**: Uses FAISS for efficient similarity search of document chunks
- **Embeddings**: HuggingFace's `sentence-transformers/all-MiniLM-L6-v2` for converting text to vectors
- **Document Processing**: Supports PDF and TXT files from `/home/msleh/gpt_project/mis_documentos`
- **Chunking Strategy**: 1000 characters per chunk with 200 character overlap
- **Retrieval**: Top-3 relevant chunks are passed as context to the LLM

Key workflow:
1. Load documents → Split into chunks → Create embeddings → Store in FAISS
2. Query → Retrieve relevant chunks → Pass to LLM with context → Generate answer

### Fine-tuning System (`finetuning.py`)
- **Training Method**: QLoRA (4-bit quantization + LoRA adapters)
- **Target Modules**: Attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **LoRA Config**: r=16, alpha=32, dropout=0.05
- **Optimization**: Uses `paged_adamw_32bit` optimized for QLoRA
- **Dataset Format**: JSONL with `instruction` and `output` fields
- **Output**: Saves trained adapters to `./gpt_oss_fine_tuned/`

Training configuration:
- Batch size: 4 per device
- Learning rate: 2e-4
- Epochs: 3
- Precision: bfloat16 (requires compatible GPU) or fp16

## Dependencies

Install required packages:
```bash
# RAG dependencies
pip install langchain-community langchain-core transformers accelerate sentence-transformers faiss-cpu pypdf

# Fine-tuning dependencies (in addition to above)
pip install datasets peft trl bitsandbytes

# For GPU acceleration (if available)
pip install faiss-gpu
```

## Running the Code

### RAG System
```bash
# Ensure Ollama is running with the model
ollama serve
# Or in a separate terminal:
ollama run gpt-oss:20b

# Run the RAG system
python chatrag_py.py
```

The script automatically:
- Attempts to start Ollama if not running
- Loads documents from `mis_documentos/`
- Creates vector store
- Enters interactive Q&A mode (type 'salir' to exit)

### Fine-tuning
```bash
python finetuning.py
```

Ensure `dataset.jsonl` exists with training examples in the format:
```json
{"instruction": "Question or task", "output": "Expected response"}
```

## Important Notes

- **Model Loading**: Both scripts expect the Ollama model named `gpt-oss:20b` to be available
- **Device Selection**: Scripts auto-detect CUDA availability and fall back to CPU
- **Memory Requirements**: Fine-tuning with 4-bit quantization significantly reduces VRAM needs
- **Document Location**: RAG documents must be placed in `mis_documentos/` folder
- **Output Directory**: Fine-tuned adapters save to `./gpt_oss_fine_tuned/`

## Key Differences Between RAG and Fine-tuning

- **RAG**: Uses external documents as context without modifying the model. Best for knowledge that changes frequently or needs citations.
- **Fine-tuning**: Permanently modifies model weights to learn new behaviors or styles. Best for adapting the model's tone, format, or specialized domain knowledge.
