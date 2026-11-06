"""Mostra como um artigo foi dividido em chunks e o score de cada um para uma consulta.

Uso:
    python debug_chunks.py
    python debug_chunks.py "daily routine of a QA" bharani_asks_about_daily_practice
"""
import sys

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_PATH, EMBEDDING_MODEL_NAME, DEVICE

query = sys.argv[1] if len(sys.argv) > 1 else "daily routine of a QA"
article = sys.argv[2] if len(sys.argv) > 2 else "bharani_asks_about_daily_practice"

emb = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=emb)

data = db.get(include=["documents", "metadatas"])
docs = [
    d for d, m in zip(data["documents"], data["metadatas"])
    if article in str(m.get("source", ""))
]

print(f"\nChunks de '{article}': {len(docs)}")
if not docs:
    sys.exit("Nenhum chunk encontrado. Confira o nome do arquivo.")

q = np.array(emb.embed_query(query))
vs = np.array(emb.embed_documents(docs))
scores = vs @ q  # vetores normalizados: produto escalar = cosseno

print(f"Consulta: {query}\n")
for d, s in sorted(zip(docs, scores), key=lambda x: -x[1]):
    preview = d.strip().replace("\n", " ")[:90]
    print(f"{s:.3f} | {len(d):5d} chars | {preview}")