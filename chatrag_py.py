# Este script crea un sistema RAG (Retrieval-Augmented Generation) para
# que tu modelo GPT-OSS responda preguntas basándose en tus propios documentos.

# --- DEPENDENCIAS ---
# Asegúrate de instalar estas bibliotecas con pip:
# pip install langchain-community langchain-core transformers accelerate sentence-transformers faiss-cpu pypdf
# Nota: "faiss-cpu" es la versión para CPU. Si tu GPU tiene suficiente VRAM,
# puedes usar "faiss-gpu" para un mejor rendimiento.

import os
import torch
import subprocess
import time
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import logging

# Configurar el registro para ver los mensajes de depuración
logging.basicConfig(level=logging.INFO)

# --- CONFIGURACIÓN ---
# Define la ruta de la carpeta donde están tus documentos.
# Usa la ruta absoluta de Linux para evitar errores.
DOCS_FOLDER = "/home/msleh/gpt_project/mis_documentos"

# El modelo de embedding a usar para convertir texto en vectores.
EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ¡IMPORTANTE!: Define el nombre del modelo tal como lo conoces en Ollama.
OLLAMA_MODEL_NAME = "gpt-oss:20b"

# --- PASO 0: INICIAR OLLAMA SI NO ESTÁ CORRIENDO ---
def start_ollama():
    """Verifica si Ollama está corriendo y lo inicia si no lo está."""
    logging.info("Verificando si el servidor de Ollama está en ejecución...")
    try:
        # Intenta obtener el estado del servicio de Ollama.
        result = subprocess.run(['pkill', '-f', 'ollama serve'], capture_output=True)
        # Si el proceso no existe, run devuelve un código de error.
        if result.returncode != 0:
            logging.info("El servidor de Ollama no está corriendo. Iniciándolo...")
            # Iniciar el servidor de Ollama en segundo plano.
            # Puedes usar 'ollama run <model_name>' o 'ollama serve'.
            # 'ollama run' es más directo si solo usarás un modelo.
            # subprocess.Popen(['ollama', 'serve'])
            subprocess.Popen(['ollama', 'run', OLLAMA_MODEL_NAME])
            time.sleep(15)  # Esperar un tiempo prudente para que el servidor se inicialice.
            logging.info("Servidor de Ollama iniciado. Esperando la carga del modelo.")
            # Un tiempo de espera para que el modelo se cargue en memoria.
            # Si el modelo es muy grande, es posible que necesites aumentar este tiempo.
            time.sleep(30)
            logging.info("Se espera que el modelo esté cargado. Continuando...")
    except FileNotFoundError:
        logging.error("Ollama no se encontró en el PATH. Asegúrate de que está instalado y configurado correctamente.")
        return False
    except Exception as e:
        logging.error(f"Ocurrió un error al intentar iniciar Ollama: {e}")
        return False
    return True

# --- PASO 1: CARGAR Y PREPARAR TUS DOCUMENTOS ---
def load_documents(folder_path):
    """Carga todos los documentos de una carpeta, admitiendo .txt y .pdf."""
    all_documents = []
    if not os.path.exists(folder_path):
        logging.info(f"Creando la carpeta '{folder_path}'. Por favor, coloca tus documentos aquí.")
        os.makedirs(folder_path)
    
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            logging.info(f"Cargando documento PDF: {filename}")
            try:
                loader = PyPDFLoader(filepath)
                all_documents.extend(loader.load())
            except Exception as e:
                logging.error(f"Error al cargar {filename}: {e}")
        elif filename.endswith(".txt"):
            logging.info(f"Cargando documento de texto: {filename}")
            try:
                loader = TextLoader(filepath)
                all_documents.extend(loader.load())
            except Exception as e:
                logging.error(f"Error al cargar {filename}: {e}")
    return all_documents

# --- PASO 2: DIVIDIR LOS DOCUMENTOS EN FRAGMENTOS (CHUNKS) ---
def split_documents(documents):
    """Divide los documentos cargados en fragmentos más pequeños."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    return text_splitter.split_documents(documents)

# --- PASO 3: CREAR INCORPORACIONES (EMBEDDINGS) ---
def create_embeddings(model_name):
    """Crea un objeto de embedding usando un modelo de Hugging Face."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Usando el dispositivo: {device}")
    model_kwargs = {"device": device}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs
    )
    return embeddings

# --- PASO 4: CREAR EL ALMACÉN DE VECTORES ---
def create_vector_store(texts, embeddings):
    """Crea y popula el almacén de vectores con los documentos fragmentados."""
    logging.info("Creando el almacén de vectores (vector store)...")
    vector_store = FAISS.from_documents(texts, embeddings)
    logging.info("Almacén de vectores creado exitosamente.")
    return vector_store

# --- PASO 5: CARGAR TU MODELO GPT-OSS ---
def load_llm(model_name):
    """Carga tu modelo GPT-OSS local usando el nombre del modelo a través de la API de Ollama."""
    logging.info(f"Conectando a Ollama y cargando el modelo: {model_name}")
    try:
        llm = Ollama(model=model_name)
        logging.info("Conexión con Ollama exitosa. Modelo listo.")
        return llm
    except Exception as e:
        logging.error(f"Error al conectar con el servidor de Ollama: {e}")
        logging.info("Asegúrate de que Ollama esté corriendo en otra terminal con 'ollama serve' o 'ollama run gpt-oss:20b'.")
        return None

# --- PASO 6: CREAR LA CADENA RAG ---
def create_rag_chain(vector_store, llm):
    """Crea la cadena de RAG que une el almacén de vectores y el LLM."""
    if not llm:
        return None
    logging.info("Creando la cadena RAG...")
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever, 
        return_source_documents=True
    )
    logging.info("Cadena RAG creada y lista.")
    return qa_chain

# --- FUNCIÓN PRINCIPAL ---
def main():
    """Ejecuta el flujo completo de RAG."""
    # Intentar iniciar Ollama si no está corriendo.
    if not start_ollama():
        logging.error("No se pudo iniciar Ollama. Saliendo del programa.")
        return

    # Primero, carga y prepara los documentos.
    documents = load_documents(DOCS_FOLDER)
    if not documents:
        logging.warning("No se encontraron documentos para procesar en la carpeta.")
        return
        
    texts = split_documents(documents)
    
    # Creamos embeddings y el almacén de vectores.
    embeddings = create_embeddings(EMBEDDINGS_MODEL_NAME)
    vector_store = create_vector_store(texts, embeddings)
    
    # Cargamos el modelo local por su nombre.
    llm = load_llm(OLLAMA_MODEL_NAME)
    
    # Creamos la cadena RAG.
    qa_chain = create_rag_chain(vector_store, llm)
    if not qa_chain:
        logging.error("No se pudo crear la cadena RAG. Saliendo.")
        return

    logging.info("\n¡El sistema RAG está listo! Puedes empezar a hacer preguntas.")
    logging.info("Escribe 'salir' para terminar.")
    
    while True:
        query = input("\nTu pregunta: ")
        if query.lower() == "salir":
            break
        
        logging.info("Buscando y generando respuesta...")
        try:
            result = qa_chain({"query": query})
            
            print("\n--- Respuesta ---")
            print(result["result"])
            
            print("\n--- Fuentes ---")
            for doc in result["source_documents"]:
                print(f"-> Fuente: {doc.metadata.get('source', 'Desconocida')}, Pág: {doc.metadata.get('page', 'Desconocida')}")
        except Exception as e:
            logging.error(f"Ocurrió un error al procesar la consulta: {e}")
            
if __name__ == "__main__":
    main()

