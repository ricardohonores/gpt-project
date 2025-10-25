# CLAUDE.md

Español | [English](CLAUDE.md)

Este archivo proporciona orientación a Claude Code (claude.ai/code) cuando trabaja con código en este repositorio.

## Descripción General del Proyecto

Este es un proyecto de experimentación con GPT basado en Python que implementa dos técnicas complementarias de IA:
1. **RAG (Retrieval-Augmented Generation)**: Consulta documentos usando búsqueda vectorial para proporcionar contexto a un LLM local
2. **Fine-tuning**: Adapta un modelo pre-entrenado usando QLoRA (Quantized Low-Rank Adaptation)

El proyecto usa Ollama para ejecutar el modelo `gpt-oss:20b` localmente y está diseñado para funcionar tanto con GPU (CUDA) como con CPU.

## Arquitectura

### Sistema RAG (`chatrag_py.py`)
- **Integración LLM**: Se conecta a la API de Ollama para usar el modelo `gpt-oss:20b`
- **Almacén Vectorial**: Usa FAISS para búsqueda eficiente de similitud de fragmentos de documentos
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` de HuggingFace para convertir texto a vectores
- **Procesamiento de Documentos**: Soporta archivos PDF y TXT desde `/home/msleh/gpt_project/mis_documentos`
- **Estrategia de Fragmentación**: 1000 caracteres por fragmento con 200 caracteres de solapamiento
- **Recuperación**: Los 3 fragmentos más relevantes se pasan como contexto al LLM

Flujo de trabajo clave:
1. Cargar documentos → Dividir en fragmentos → Crear embeddings → Almacenar en FAISS
2. Consulta → Recuperar fragmentos relevantes → Pasar al LLM con contexto → Generar respuesta

### Sistema de Fine-tuning (`finetuning.py`)
- **Método de Entrenamiento**: QLoRA (cuantización de 4 bits + adaptadores LoRA)
- **Módulos Objetivo**: Capas de atención (`q_proj`, `k_proj`, `v_proj`, `o_proj`)
- **Configuración LoRA**: r=16, alpha=32, dropout=0.05
- **Optimización**: Usa `paged_adamw_32bit` optimizado para QLoRA
- **Formato del Dataset**: JSONL con campos `instruction` y `output`
- **Salida**: Guarda los adaptadores entrenados en `./gpt_oss_fine_tuned/`

Configuración de entrenamiento:
- Tamaño del lote: 4 por dispositivo
- Tasa de aprendizaje: 2e-4
- Épocas: 3
- Precisión: bfloat16 (requiere GPU compatible) o fp16

## Dependencias

Instalar paquetes requeridos:
```bash
# Dependencias para RAG
pip install langchain-community langchain-core transformers accelerate sentence-transformers faiss-cpu pypdf

# Dependencias para fine-tuning (además de las anteriores)
pip install datasets peft trl bitsandbytes

# Para aceleración GPU (si está disponible)
pip install faiss-gpu
```

## Ejecución del Código

### Sistema RAG
```bash
# Asegúrate de que Ollama esté corriendo con el modelo
ollama serve
# O en una terminal separada:
ollama run gpt-oss:20b

# Ejecuta el sistema RAG
python chatrag_py.py
```

El script automáticamente:
- Intenta iniciar Ollama si no está corriendo
- Carga documentos desde `mis_documentos/`
- Crea el almacén vectorial
- Entra en modo interactivo de preguntas y respuestas (escribe 'salir' para terminar)

### Fine-tuning
```bash
python finetuning.py
```

Asegúrate de que `dataset.jsonl` existe con ejemplos de entrenamiento en el formato:
```json
{"instruction": "Pregunta o tarea", "output": "Respuesta esperada"}
```

## Notas Importantes

- **Carga del Modelo**: Ambos scripts esperan que el modelo de Ollama llamado `gpt-oss:20b` esté disponible
- **Selección de Dispositivo**: Los scripts detectan automáticamente la disponibilidad de CUDA y recurren a CPU
- **Requisitos de Memoria**: El fine-tuning con cuantización de 4 bits reduce significativamente las necesidades de VRAM
- **Ubicación de Documentos**: Los documentos para RAG deben colocarse en la carpeta `mis_documentos/`
- **Directorio de Salida**: Los adaptadores fine-tuned se guardan en `./gpt_oss_fine_tuned/`

## Diferencias Clave entre RAG y Fine-tuning

- **RAG**: Usa documentos externos como contexto sin modificar el modelo. Mejor para conocimiento que cambia frecuentemente o necesita citaciones.
- **Fine-tuning**: Modifica permanentemente los pesos del modelo para aprender nuevos comportamientos o estilos. Mejor para adaptar el tono del modelo, formato o conocimiento especializado de dominio.
