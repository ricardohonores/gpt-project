# GPT Project: RAG & Fine-tuning

Español | [English](README.md)

Proyecto de experimentación con técnicas avanzadas de IA: **RAG (Retrieval-Augmented Generation)** y **Fine-tuning** usando modelos de lenguaje locales.

## 🎯 Descripción

Este proyecto implementa dos enfoques complementarios para mejorar las capacidades de modelos de lenguaje:

1. **Sistema RAG**: Permite que un LLM local responda preguntas basándose en documentos propios usando búsqueda vectorial
2. **Fine-tuning con QLoRA**: Adapta un modelo pre-entrenado a tareas específicas usando cuantización de 4 bits

## ✨ Características

### Sistema RAG (`chatrag_py.py`)
- 📚 Procesa documentos PDF y TXT
- 🔍 Búsqueda semántica con FAISS
- 🤖 Integración con Ollama para inferencia local
- 💬 Interfaz interactiva de preguntas y respuestas
- 📝 Muestra fuentes de información en cada respuesta

### Fine-tuning (`finetuning.py`)
- ⚡ QLoRA (Quantized Low-Rank Adaptation) para entrenar con poca VRAM
- 🎯 Cuantización de 4 bits
- 💾 Guarda solo adaptadores (pesos ligeros)
- 🔧 Configurable para GPU o CPU

## 📋 Requisitos

- Python 3.8+
- [Ollama](https://ollama.ai/) instalado y corriendo
- Modelo `gpt-oss:20b` descargado en Ollama
- GPU con CUDA (opcional, pero recomendado para fine-tuning)

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/ricardohonores/gpt-project.git
cd gpt-project
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias

**Para el sistema RAG:**
```bash
pip install langchain-community langchain-core transformers accelerate sentence-transformers faiss-cpu pypdf
```

**Para fine-tuning (adicional):**
```bash
pip install datasets peft trl bitsandbytes
```

**Para GPU (opcional):**
```bash
pip install faiss-gpu
```

### 4. Instalar y configurar Ollama
```bash
# Instalar Ollama (si no lo tienes)
curl -fsSL https://ollama.ai/install.sh | sh

# Descargar el modelo
ollama pull gpt-oss:20b
```

## 💻 Uso

### Sistema RAG

1. **Coloca tus documentos** en la carpeta `mis_documentos/` (archivos .txt o .pdf)

2. **Inicia Ollama** (en otra terminal):
```bash
ollama serve
```

3. **Ejecuta el sistema RAG**:
```bash
python chatrag_py.py
```

4. **Haz preguntas** interactivamente. Escribe `salir` para terminar.

**Ejemplo:**
```
Tu pregunta: ¿Qué es la IA generativa?

--- Respuesta ---
La IA generativa es una inteligencia artificial que crea contenido nuevo...

--- Fuentes ---
-> Fuente: /path/to/document.pdf, Pág: 5
```

### Fine-tuning

1. **Prepara tu dataset** en formato JSONL (`dataset.jsonl`):
```json
{"instruction": "Pregunta o tarea", "output": "Respuesta esperada"}
{"instruction": "Otra pregunta", "output": "Otra respuesta"}
```

2. **Ejecuta el entrenamiento**:
```bash
python finetuning.py
```

3. **Los adaptadores se guardan** en `./gpt_oss_fine_tuned/`

## 📁 Estructura del Proyecto

```
gpt-project/
├── chatrag_py.py          # Sistema RAG completo
├── finetuning.py          # Script de fine-tuning con QLoRA
├── dataset.jsonl          # Dataset de ejemplo
├── CLAUDE.md              # Documentación técnica (inglés)
├── CLAUDE.es.md           # Documentación técnica (español)
├── README.md              # README en inglés
├── README.es.md           # Este archivo (español)
├── .gitignore             # Archivos ignorados por git
└── mis_documentos/        # Carpeta de documentos para RAG
    ├── *.pdf
    └── *.txt
```

## 🔧 Configuración

### Sistema RAG
Edita las siguientes variables en `chatrag_py.py`:
```python
DOCS_FOLDER = "/ruta/a/tus/documentos"
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME = "gpt-oss:20b"
```

### Fine-tuning
Edita las siguientes variables en `finetuning.py`:
```python
OLLAMA_MODEL_NAME = "gpt-oss:20b"
DATASET_FILE = "dataset.jsonl"
OUTPUT_DIR = "./gpt_oss_fine_tuned"
```

## 🤔 ¿RAG o Fine-tuning?

| Característica | RAG | Fine-tuning |
|----------------|-----|-------------|
| **Modifica el modelo** | ❌ No | ✅ Sí |
| **Requiere entrenamiento** | ❌ No | ✅ Sí |
| **Documentos dinámicos** | ✅ Sí | ❌ No |
| **Muestra fuentes** | ✅ Sí | ❌ No |
| **Cambia comportamiento** | ❌ No | ✅ Sí |
| **Uso de memoria** | 🟢 Bajo | 🟡 Medio |

**Usa RAG cuando:** Necesites responder preguntas sobre documentos específicos o conocimiento que cambia frecuentemente.

**Usa Fine-tuning cuando:** Quieras que el modelo aprenda un estilo específico, dominio técnico o nuevos patrones de comportamiento.

## 📚 Documentación Adicional

Para detalles técnicos sobre la arquitectura y configuración, consulta [CLAUDE.es.md](CLAUDE.es.md).

## 📄 Licencia

Este proyecto es de código abierto y está disponible para uso educativo y experimental.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Siéntete libre de abrir issues o pull requests.
