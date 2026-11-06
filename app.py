import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from rag_core import (
    BACKEND, CLARIFY_MESSAGE, MAX_TURNS, NO_DOCS_MESSAGE, SetupError, clean,
    describe_docs, load_pipeline, source_entries, sources_as_markdown, strip_markers,
)

st.set_page_config(page_title="Assistente de Testes", page_icon="🧪")
st.title("🧪 Assistente de Testes")
st.caption("Respostas baseadas nos artigos de James Bach (satisfice.com)")


@st.cache_resource(show_spinner="Carregando modelos (só na primeira vez)...")
def get_pipeline():
    return load_pipeline()


try:
    pipeline = get_pipeline()
except SetupError as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []   # o que é exibido na tela
if "history" not in st.session_state:
    st.session_state.history = []    # memória enviada ao LLM

with st.sidebar:
    st.header("Opções")
    show_details = st.toggle("Mostrar detalhes da busca", value=False)
    if st.button("Limpar conversa"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()
    st.caption(f"LLM: {BACKEND}")


def render_details(record):
    if not show_details or not record.get("queries"):
        return
    with st.expander("Detalhes da busca"):
        st.markdown("**Consultas geradas**")
        for q in record["queries"]:
            st.text(q)
        st.markdown("**Trechos recuperados**")
        for i, line in enumerate(record["chunks"], 1):
            st.text(f"[{i}] {line}")


for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources_md"):
            st.markdown(m["sources_md"])
        if m["role"] == "assistant":
            render_details(m)

question = st.chat_input("Pergunte sobre os artigos...")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    history = st.session_state.history
    with st.chat_message("assistant"):
        try:
            with st.spinner("Buscando nos artigos..."):
                queries, docs = pipeline.prepare(question, history)

            record = {"role": "assistant", "content": "", "sources_md": "",
                      "queries": queries or [], "chunks": describe_docs(docs)}
            keep_in_memory = True

            if queries is None:
                text = CLARIFY_MESSAGE
                keep_in_memory = False
                st.markdown(text)
            elif not docs:
                text = NO_DOCS_MESSAGE
                st.markdown(text)
            else:
                raw = st.write_stream(pipeline.stream_answer(question, history, docs))
                text = clean(raw)
                entries = source_entries(text, docs)
                if entries:
                    record["sources_md"] = sources_as_markdown(entries)
                    st.markdown(record["sources_md"])

            record["content"] = text
            render_details(record)
        except Exception as e:  # ex.: Ollama caiu no meio da resposta
            st.error(f"Não consegui gerar a resposta: {e}")
            st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append(record)
    if keep_in_memory:
        history += [HumanMessage(content=question), AIMessage(content=strip_markers(text))]
        st.session_state.history = history[-2 * MAX_TURNS:]