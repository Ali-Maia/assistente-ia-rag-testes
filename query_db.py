import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter
from config import CHROMA_PATH, EMBEDDING_MODEL_NAME, DEVICE

# Evita alertas secundários de paralelismo da Hugging Face
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def build_retriever(
    k: int = 6,
    fetch_k: int = 20,
    similarity_threshold: float = 0.5,
    lambda_mult: float = 0.85,
):
    """Monta o retriever (MMR + filtro por similaridade) uma única vez.

    O limiar de 0.5 foi calibrado com o debug_scores.py para consultas em
    INGLÊS (relevantes: ~0.57-0.65; fora do assunto: ~0.44). O chatbot
    reescreve as perguntas para inglês antes de buscar.
    """

    # 1. Modelo de Embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 2. Conectar ao Banco Vetorial
    vectorstore = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_model,
    )

    # 3. Retriever Base com MMR (Busca Semântica + Diversidade)
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )

    # 4. Compressor de Contexto (Filtra ruídos por limiar de similaridade)
    embeddings_filter = EmbeddingsFilter(
        embeddings=embedding_model,
        similarity_threshold=similarity_threshold,
    )

    return ContextualCompressionRetriever(
        base_compressor=embeddings_filter,
        base_retriever=base_retriever,
    )


def search_articles(query: str, k: int = 6):
    """Função de teste: busca e imprime os trechos, sem gerar resposta."""
    print(f"Embeddings rodando em: {DEVICE.upper()}")

    retriever = build_retriever(k=k)

    print(f"\nBuscando trechos relevantes com MMR + Compressão para: '{query}'\n")
    compressed_docs = retriever.invoke(query)

    if not compressed_docs:
        print("Nenhum trecho passou pelo filtro de similaridade.")
        return

    for idx, doc in enumerate(compressed_docs, 1):
        fonte = doc.metadata.get("source", "Desconhecido")
        header = doc.metadata.get("Header 1", doc.metadata.get("Header 2", "N/A"))
        print(f"--- [Resultado Comprimido #{idx}] ---")
        print(f"Arquivo: {fonte}")
        print(f"Seção: {header}")
        print(f"Conteúdo:\n{doc.page_content.strip()}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    query = "What are the core elements of Rapid Software Testing?"
    search_articles(query, k=6)