import os
import re
import shutil
import sys

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import ARTICLES_DIR, CHROMA_PATH, EMBEDDING_MODEL_NAME, DEVICE

# O bge-m3 aceita até 8192 tokens, então chunks maiores não são problema.
# Eles mantêm as ideias de um trecho juntas (ex.: a lista de práticas diárias).
CHUNK_SIZE = 1800
CHUNK_OVERLAP = 250
MIN_CHUNK_CHARS = 150  # descarta fragmentos minúsculos (só título, separadores etc.)

FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_front_matter(text: str):
    """Separa o bloco '--- title/author/date/url ---' do corpo do artigo."""
    meta = {}
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return meta, text
    for line in match.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep:
            meta[key.strip()] = value.strip().strip('"')
    return meta, text[match.end():]


def extract_year(date_text: str):
    match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", date_text or "")
    return int(match.group(1)) if match else None


def clean_body(text: str) -> str:
    """Remove ruído que se repete em todo artigo e polui os embeddings."""
    text = re.sub(r"^\*\*(?:Fonte Original|Data):\*\*.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*-{3,}\s*$", "", text, flags=re.MULTILINE)  # linhas '---'
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_chunks(documents):
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = []
    for doc in documents:
        meta, body = parse_front_matter(doc.page_content)
        body = clean_body(body)

        source = doc.metadata.get("source", "")
        title = meta.get("title") or os.path.splitext(os.path.basename(source))[0]
        author = meta.get("author")

        # Metadados extras (o Chroma só aceita str/int/float/bool, nunca None)
        extra = {"title": title}
        if author:
            extra["author"] = author
        if meta.get("url"):
            extra["url"] = meta["url"]
        year = extract_year(meta.get("date", ""))
        if year:
            extra["year"] = year

        # Todo chunk começa com o título: assim, mesmo um trecho que fala só de
        # "explique o teste todo dia" carrega o contexto "Daily Practice" no embedding.
        prefix = f"Article: {title}" + (f" (by {author})" if author else "")

        for sub_doc in markdown_splitter.split_text(body):
            for split in text_splitter.split_documents([sub_doc]):
                if len(split.page_content.strip()) < MIN_CHUNK_CHARS:
                    continue
                split.metadata.update(doc.metadata)
                split.metadata.update(extra)
                split.page_content = f"{prefix}\n\n{split.page_content}"
                chunks.append(split)

    return chunks


def build_vector_store():
    print(f"Dispositivo em execução: {DEVICE.upper()}")

    # 1. Carregar documentos
    loader = DirectoryLoader(
        ARTICLES_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"Artigos carregados na memória: {len(documents)}")

    # 2. Gerar chunks
    chunks = build_chunks(documents)
    avg = sum(len(c.page_content) for c in chunks) // max(len(chunks), 1)
    print(f"Chunks gerados: {len(chunks)} (média de {avg} caracteres)")

    # 3. Apagar o banco antigo. O Chroma.from_documents ADICIONA ao que já existe,
    #    então sem isso os chunks velhos ficariam misturados com os novos.
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            print(f"Banco antigo removido: '{CHROMA_PATH}'")
        except PermissionError:
            sys.exit("Não consegui apagar o banco. Feche o chatbot.py (ou outro "
                     "processo que esteja usando o Chroma) e rode de novo.")

    # 4. Configurar Embedding
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": DEVICE},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 32 if DEVICE == "cuda" else 16},
    )
    print(f"Modelo {EMBEDDING_MODEL_NAME} carregado!")

    # 5. Persistir no ChromaDB
    print("Iniciando indexação no ChromaDB...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=CHROMA_PATH,
    )
    print(f"Banco vetorial criado com sucesso em '{CHROMA_PATH}'!")
    return vectorstore


if __name__ == "__main__":
    build_vector_store()