import os
import torch

ARTICLES_DIR = "satisfice_articles"
CHROMA_PATH = "chroma_db_james_bach"
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"

# Define o dispositivo automaticamente
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"