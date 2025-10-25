import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    set_seed
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import logging
from functools import partial


# --- Configurar el logging para ver los mensajes ---
logging.basicConfig(level=logging.INFO)


# --- Configuración del modelo y del entrenamiento ---
# Define el nombre de tu modelo en Ollama
OLLAMA_MODEL_NAME = "gpt-oss:20b"
# Define la ruta de tu dataset
DATASET_FILE = "dataset.jsonl"
# Nombre de la carpeta donde se guardarán los adaptadores (los pesos entrenados)
OUTPUT_DIR = "./gpt_oss_fine_tuned"


# --- 1. Cargar el Tokenizer y el Modelo (con Cuantización 4-bits) ---
logging.info("Iniciando la carga del modelo base y tokenizer con cuantización 4-bits...")


# Configuración de BitsAndBytes para cuantización (QLoRA)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=False,
)


# Cargar el modelo base en 4-bits.
# Usamos 'device_map="auto"' para que el modelo se distribuya en la GPU/CPU automáticamente.
model = AutoModelForCausalLM.from_pretrained(
    OLLAMA_MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto"
)


# El tokenizer es crucial para que el modelo entienda el texto.
tokenizer = AutoTokenizer.from_pretrained(OLLAMA_MODEL_NAME)
# Si el modelo no tiene un token de relleno (pad token), lo añadimos.
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer)) # Ajustamos los embeddings del modelo al nuevo token.


# --- 2. Preparar el Modelo para QLoRA ---
logging.info("Preparando el modelo para QLoRA...")


# Habilitar el gradiente para los embeddings de los adaptadores
model.config.use_cache = False
model.config.pretraining_tp = 1
model = prepare_model_for_kbit_training(model)


# Configuración de PEFT (Parameter-Efficient Fine-Tuning)
# Esta es la configuración de LoRA, que define qué partes del modelo se entrenarán.
peft_config = LoraConfig(
    r=16, # Rango de la matriz de adaptadores. Un valor más alto significa más capacidad de aprendizaje.
    lora_alpha=32, # Factor de escalado de los adaptadores.
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # Módulos de atención a entrenar.
    lora_dropout=0.05, # Tasa de dropout para regularización.
    bias="none", # No entrenar los sesgos.
    task_type="CAUSAL_LM", # Tipo de tarea: Modelado de lenguaje causal.
)


# Obtener el modelo PEFT (Parameter-Efficient Fine-Tuning)
model = get_peft_model(model, peft_config)


# --- 3. Cargar y Pre-procesar el Dataset ---
logging.info("Cargando y pre-procesando el dataset...")


# Cargar el dataset desde el archivo JSONL.
dataset = load_dataset("json", data_files=DATASET_FILE, split="train")


# Formatear el dataset en un formato de prompt estándar para el entrenamiento.
# Esto asegura que el modelo vea los datos de una forma consistente.
def formatting_prompts_func(example):
    text = f"### Instruction: {example['instruction']}\n### Output: {example['output']}"
    return {"text": text}


formatted_dataset = dataset.map(formatting_prompts_func)


# --- 4. Configurar y Ejecutar el Entrenador (Trainer) ---
logging.info("Configurando el SFTTrainer...")


# Configuración de los argumentos de entrenamiento.
training_arguments = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4, # Tamaño del lote por dispositivo.
    gradient_accumulation_steps=1, # Pasos de acumulación de gradiente.
    optim="paged_adamw_32bit", # Optimizador optimizado para QLoRA.
    learning_rate=2e-4, # Tasa de aprendizaje.
    num_train_epochs=3, # Número de épocas de entrenamiento.
    fp16=False, # Si tu GPU no es compatible con bf16, usa fp16.
    bf16=True, # Usar bfloat16 si la GPU lo soporta (NVIDIA A100/A3000+).
    logging_steps=10, # Imprimir el log cada 10 pasos.
    save_strategy="epoch", # Guardar el modelo en cada época.
    push_to_hub=False, # No subir el modelo a Hugging Face Hub.
)


# Configurar el entrenador SFT (Supervised Fine-Tuning)
trainer = SFTTrainer(
    model=model,
    train_dataset=formatted_dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=512, # Longitud máxima de la secuencia de texto.
    tokenizer=tokenizer,
    args=training_arguments,
)


logging.info("Entrenamiento iniciado...")
# Iniciar el entrenamiento
trainer.train()


# --- 5. Guardar el Modelo ---
logging.info("Entrenamiento finalizado. Guardando el modelo y el tokenizer...")


# Guardar los adaptadores (pequeños pesos entrenados).
trainer.model.save_pretrained(OUTPUT_DIR)


# Guardar el tokenizer también
tokenizer.save_pretrained(OUTPUT_DIR)


logging.info("¡Fine-tuning completado! Los adaptadores se guardaron en la carpeta: %s", OUTPUT_DIR)
logging.info("Ahora puedes usar estos adaptadores para cargar tu modelo afinado.")
