import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import CHROMA_PATH, EMBEDDING_MODEL_NAME, DEVICE

emb = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": DEVICE},
    encode_kwargs={"normalize_embeddings": True},
)
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=emb)

queries = [
    "quais os hábitos de um bom qa",
    "dia a dia de um qa",
    "como fazer checking",
    "Qual é a capital da França?",
    "Explique o conceito de testability",
    "habits of a good QA tester",
    "daily routine of a QA tester",
    "how to do checking",
    "What is the capital of France?",
]

for q in queries:
    qv = np.array(emb.embed_query(q))
    docs = db.similarity_search(q, k=5)
    dvs = np.array(emb.embed_documents([d.page_content for d in docs]))
    scores = dvs @ qv  # vetores normalizados: produto escalar = cosseno
    print(f"\n### {q}")
    for d, s in zip(docs, scores):
        print(f"{s:.3f}  {d.metadata.get('source', '?')}")