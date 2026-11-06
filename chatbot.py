import os
import re
import sys

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from query_db import build_retriever

MAX_TURNS = 3       # quantas trocas (pergunta + resposta) ficam na memória
MAX_QUERIES = 3     # variações de busca para perguntas amplas
K_PER_QUERY = 5     # trechos recuperados por consulta
MAX_DOCS = 10       # máximo de trechos enviados ao LLM
SHOW_QUERY = True   # mostra consultas e trechos recuperados (depuração)

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
     "como [2], logo após a informação. Não escreva nomes de arquivos nem uma "
     "lista de fontes no final: o programa a adiciona. Se não houver resposta "
     "no contexto, não cite nada.\n\n"
     "Contexto:\n{context}"),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def clean(text: str) -> str:
    """Remove blocos <think> (modelos Qwen3), aspas e espaços sobrando."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    return text.strip().strip("\"'").strip()


def short_name(source: str) -> str:
    return re.split(r"[\\/]", source or "?")[-1]


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
        f"[{i}] (fonte: {short_name(d.metadata.get('source', '?'))})\n{d.page_content.strip()}"
        for i, d in enumerate(docs, 1)
    )


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


def add_sources(answer: str, docs) -> str:
    """Converte as citações [n] do modelo em uma lista de arquivos, montada pelo código."""
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(docs)})
    if not used:
        return answer
    lines = [f"  [{n}] {short_name(docs[n - 1].metadata.get('source', '?'))}" for n in used]
    return answer + "\n\nFontes:\n" + "\n".join(lines)


def main():
    # Falha rápido (antes de carregar o bge-m3) se o servidor do Ollama estiver parado
    try:
        httpx.get("http://localhost:11434", timeout=3)
    except httpx.HTTPError:
        sys.exit("O Ollama não está rodando. Abra o app do Ollama (ícone na bandeja) "
                 "ou rode 'ollama serve' em outro terminal e tente de novo.")

    print("Carregando modelos...")
    retriever = build_retriever(k=K_PER_QUERY, fetch_k=30)
    llm = ChatOllama(model="qwen3:8b", temperature=0.2, num_ctx=8192)

    rewrite_chain = REWRITE_PROMPT | llm | StrOutputParser()
    answer_chain = ANSWER_PROMPT | llm | StrOutputParser()

    history = []
    print("Chatbot pronto. Digite 'sair' para encerrar ou '/limpar' para zerar a memória.\n")

    while True:
        question = input("Você: ").strip()
        if question.lower() in {"sair", "exit", "quit"}:
            break
        if question.lower() == "/limpar":
            history = []
            print("Memória limpa.\n")
            continue
        if not question:
            continue

        # 1. Reescreve a pergunta (usa o histórico, traduz para inglês, 1 a 3 variações)
        raw = rewrite_chain.invoke({"history": format_history(history), "question": question})
        if "NEEDS_CLARIFICATION" in raw:
            print("\nBot: Não entendi a que você se refere. "
                  "Pode reformular a pergunta com mais detalhes?\n")
            continue
        queries = parse_queries(raw, fallback=question)
        if SHOW_QUERY:
            for q in queries:
                print(f"  [busca: {q}]")

        # 2. Recupera os trechos (união das consultas, sem duplicatas)
        docs = retrieve_docs(retriever, queries)
        if SHOW_QUERY:
            for i, d in enumerate(docs, 1):
                trecho = d.page_content.split("\n\n", 1)[-1][:60].replace("\n", " ")
                print(f"  [{i}] {short_name(d.metadata.get('source', '?'))} | {trecho}")

        # 3. Gera a resposta
        if not docs:
            answer = "Não encontrei trechos relevantes nos documentos."
            shown = answer
        else:
            answer = clean(answer_chain.invoke({
                "context": format_docs(docs),
                "history": history,
                "question": question,
            }))
            shown = add_sources(answer, docs)

        print(f"\nBot: {shown}\n")

        # 4. Atualiza a memória (sem as marcas [n], que só valem para esta rodada)
        stored = re.sub(r"\s*\[\d+\]", "", answer)
        history += [HumanMessage(content=question), AIMessage(content=stored)]
        history = history[-2 * MAX_TURNS:]


if __name__ == "__main__":
    main()