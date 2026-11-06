"""Núcleo do assistente (RAG): usado tanto pelo chatbot de terminal quanto pela interface gráfica."""
import os
import re

import httpx
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from query_db import build_retriever

MAX_TURNS = 3       # quantas trocas (pergunta + resposta) ficam na memória
MAX_QUERIES = 3     # variações de busca para perguntas amplas
K_PER_QUERY = 5     # trechos recuperados por consulta
MAX_DOCS = 10       # máximo de trechos enviados ao LLM
BACKEND = os.getenv("LLM_BACKEND", "ollama").lower()  # "ollama" (local) ou "groq" (API)

NO_DOCS_MESSAGE = "Não encontrei trechos relevantes nos documentos."
CLARIFY_MESSAGE = "Não entendi a que você se refere. Pode reformular a pergunta com mais detalhes?"


class SetupError(Exception):
    """Problema de configuração (ex.: Ollama parado) com mensagem pronta para o usuário."""


# Passo 1: transforma a mensagem do usuário em 1 a 3 consultas de busca
# autônomas e em inglês (os artigos estão em inglês).
REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You rewrite user messages into search queries for a collection of "
     "English articles about software testing.\n"
     "Rules:\n"
     "- If the message is already self-contained, just translate it to English "
     "and keep its meaning. Do NOT add topics, methodologies or product names "
     "the user did not mention, even if they appeared earlier in the conversation.\n"
     "- Use the conversation history ONLY when the message depends on it "
     "(e.g. 'isso', 'o primeiro', 'it', 'that', 'the second point'); in that "
     "case, resolve the reference.\n"
     "- If the message depends on something that cannot be identified from the "
     "history (for example 'resuma isso' with an empty history), output exactly: "
     "NEEDS_CLARIFICATION\n"
     "- For a narrow, specific question output ONE query. For a broad or "
     "open-ended question (routines, habits, overviews, 'what does X involve') "
     "output up to 3 queries, one per line, each covering a DIFFERENT angle "
     "with different keywords (for example practices, responsibilities, skills).\n"
     "- Keep each query short and keyword-rich. Do NOT answer the question.\n"
     "- Output ONLY the queries, one per line, with no numbering, quotes or "
     "explanation (or NEEDS_CLARIFICATION)."),
    ("human",
     "Conversation history:\n{history}\n\n"
     "Latest user message: {question}\n\n"
     "English search queries:"),
])

# Passo 2: responde usando apenas o contexto recuperado.
ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Você é um assistente que responde perguntas sobre teste de software "
     "usando APENAS o contexto fornecido.\n"
     "Regras:\n"
     "- Nunca invente exemplos, definições, números ou citações. Se o contexto "
     "não trouxer o que foi pedido, diga isso claramente.\n"
     "- Não complete a resposta com conhecimento próprio nem com inferências. "
     "Se o contexto cobrir o tema só em parte, avise em UMA frase que a "
     "cobertura é parcial. Não liste o que 'também faria parte' do tema e "
     "não afirme o que é comum ou normal na prática profissional.\n"
     "- Use o histórico da conversa apenas para entender a que a pergunta se "
     "refere; os fatos devem vir do contexto.\n"
     "- Os artigos costumam citar (linhas que começam com '>') opiniões de "
     "OUTRAS pessoas que o autor depois critica. Nunca apresente essas citações "
     "como fato ou como a posição do autor: diga que o autor cita aquela visão "
     "e indique se ele concorda ou discorda.\n"
     "- Use somente os trechos que respondem diretamente à pergunta; ignore "
     "trechos sobre outros assuntos, mesmo que estejam no contexto.\n"
     "- Responda no mesmo idioma da pergunta.\n"
     "- Cite o número do trecho que sustenta cada informação, entre colchetes, "
     "como [2], logo após a informação. Não escreva nomes de arquivos, links nem "
     "uma lista de fontes no final: o programa a adiciona. Se não houver resposta "
     "no contexto, não cite nada.\n\n"
     "Contexto:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


# ---------------------------------------------------------------- utilitários

def clean(text: str) -> str:
    """Remove blocos <think> (modelos Qwen3), aspas e espaços sobrando."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip().strip("\"'").strip()


def strip_markers(text: str) -> str:
    """Remove as marcas [n], que só valem para a rodada em que foram geradas."""
    return re.sub(r"\s*\[\d+\]", "", text)


def short_name(source: str) -> str:
    return re.split(r"[\\/]", source or "?")[-1]


def doc_title(doc) -> str:
    return doc.metadata.get("title") or short_name(doc.metadata.get("source", "?"))


def format_history(history) -> str:
    if not history:
        return "(empty)"
    lines = []
    for msg in history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        lines.append(f"{role}: {msg.content[:300]}")
    return "\n".join(lines)


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{i}] (fonte: {doc_title(d)})\n{d.page_content.strip()}"
        for i, d in enumerate(docs, 1)
    )


def describe_docs(docs):
    """Uma linha por trecho (arquivo + começo do texto), para depuração."""
    lines = []
    for d in docs:
        trecho = d.page_content.split("\n\n", 1)[-1][:60].replace("\n", " ")
        lines.append(f"{short_name(d.metadata.get('source', '?'))} | {trecho}")
    return lines


def parse_queries(raw: str, fallback: str):
    """Transforma a saída do reescritor em uma lista de 1 a MAX_QUERIES consultas."""
    queries = []
    for line in clean(raw).splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip(" \"'")
        if line and line not in queries:
            queries.append(line)
    return queries[:MAX_QUERIES] or [fallback]


def retrieve_docs(retriever, queries):
    """Busca com cada consulta e junta os resultados sem duplicatas.

    A união é intercalada (1º de cada consulta, 2º de cada consulta...) para que
    todas as variações contribuam, mesmo quando o limite MAX_DOCS corta o final.
    """
    results = [retriever.invoke(q) for q in queries]
    docs, seen = [], set()
    for rank in range(max(len(r) for r in results)):
        for r in results:
            if rank < len(r):
                d = r[rank]
                key = (d.metadata.get("source"), d.page_content[:200])
                if key not in seen:
                    seen.add(key)
                    docs.append(d)
    return docs[:MAX_DOCS]


# --------------------------------------------------------------------- fontes

def source_entries(answer: str, docs):
    """Agrupa os trechos citados ([n]) por artigo, com título e URL (se existir)."""
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(docs)})
    grouped = {}
    for n in used:
        d = docs[n - 1]
        title, url = doc_title(d), d.metadata.get("url")
        entry = grouped.setdefault(url or title, {"numbers": [], "title": title, "url": url})
        entry["numbers"].append(n)
    return list(grouped.values())


def sources_as_text(entries) -> str:
    lines = []
    for e in entries:
        nums = ", ".join(str(n) for n in e["numbers"])
        suffix = f" — {e['url']}" if e["url"] else ""
        lines.append(f"  [{nums}] {e['title']}{suffix}")
    return "Fontes:\n" + "\n".join(lines)


def sources_as_markdown(entries) -> str:
    lines = []
    for e in entries:
        nums = ", ".join(str(n) for n in e["numbers"])
        label = f"\\[{nums}\\]"
        if e["url"]:
            title = e["title"].replace("[", "(").replace("]", ")")
            url = e["url"].replace(" ", "%20").replace(")", "%29")
            lines.append(f"- {label} [{title}]({url})")
        else:
            lines.append(f"- {label} {e['title']}")
    return "**Fontes**\n" + "\n".join(lines)


# ------------------------------------------------------------------- pipeline

def check_ollama():
    try:
        httpx.get("http://localhost:11434", timeout=3)
    except httpx.HTTPError:
        raise SetupError("O Ollama não está rodando. Abra o app do Ollama (ícone na bandeja) "
                         "ou rode 'ollama serve' em outro terminal e tente de novo.")


def build_llm():
    """Escolhe o LLM pela variável de ambiente LLM_BACKEND ('ollama' ou 'groq')."""
    if BACKEND == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise SetupError("O backend 'groq' precisa do pacote langchain-groq: pip install langchain-groq")
        if not os.getenv("GROQ_API_KEY"):
            raise SetupError("Defina a variável GROQ_API_KEY antes de usar o backend 'groq'.")
        return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2)
    return ChatOllama(model="qwen3:8b", temperature=0.2, num_ctx=8192)


class Pipeline:
    def __init__(self, retriever, rewrite_chain, answer_chain):
        self.retriever = retriever
        self.rewrite_chain = rewrite_chain
        self.answer_chain = answer_chain

    def prepare(self, question, history):
        """Reescreve a pergunta e recupera os trechos.

        Retorna (consultas, trechos). Se a pergunta depende de algo que o histórico
        não esclarece, retorna (None, []).
        """
        raw = self.rewrite_chain.invoke({"history": format_history(history), "question": question})
        if "NEEDS_CLARIFICATION" in raw:
            return None, []
        queries = parse_queries(raw, fallback=question)
        return queries, retrieve_docs(self.retriever, queries)

    def _payload(self, question, history, docs):
        return {"context": format_docs(docs), "history": history, "question": question}

    def answer(self, question, history, docs) -> str:
        return clean(self.answer_chain.invoke(self._payload(question, history, docs)))

    def stream_answer(self, question, history, docs):
        """Gera a resposta aos poucos (para exibir na interface enquanto o modelo escreve)."""
        yield from self.answer_chain.stream(self._payload(question, history, docs))


def load_pipeline() -> Pipeline:
    if BACKEND == "ollama":
        check_ollama()  # falha rápido, antes de carregar o bge-m3
    retriever = build_retriever(k=K_PER_QUERY, fetch_k=30)
    llm = build_llm()
    return Pipeline(
        retriever=retriever,
        rewrite_chain=REWRITE_PROMPT | llm | StrOutputParser(),
        answer_chain=ANSWER_PROMPT | llm | StrOutputParser(),
    )