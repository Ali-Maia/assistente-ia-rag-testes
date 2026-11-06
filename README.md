# Assistente RAG de Testes (James Bach)

Assistente de Inteligencia Artificial especializado em teste de software, com base exclusiva no acervo de artigos de James Bach (Satisfice.com).

O projeto usa arquitetura RAG (Retrieval-Augmented Generation) com foco em grounding: a resposta é gerada a partir dos trechos recuperados do acervo, com citações numéricas rastreáveis.

## Visao Geral

- Objetivo: responder perguntas sobre teste de software com base documental.
- Base de conhecimento: artigos em Markdown no diretório `satisfice_articles/` (não versionado no Git).
- Pipeline: reescrita de pergunta -> recuperação semântica -> compressão de contexto -> resposta com citações.
- Interface principal: Streamlit com streaming de resposta em tempo real.
- Backends de LLM:
  - Local e privado via Ollama (`qwen3:8b`).
  - Nuvem via Groq (`llama-3.3-70b-versatile`).

## Principais Funcionalidades

- Arquitetura anti-alucinacao:
  - O prompt de resposta instrui o modelo a usar apenas o contexto recuperado.
  - Quando não há cobertura suficiente, o assistente deve declarar isso explicitamente.

- Recuperação avançada (Advanced RAG):
  - Chunking semântico com preservação de hierarquia de cabeçalhos Markdown.
  - Retriever vetorial com MMR para diversidade de trechos.
  - Contextual Compression com filtro de similaridade para remover ruído.

- Query rewriting:
  - Tradução automática da pergunta para inglês (acervo em inglês).
  - Decomposição de perguntas amplas em até 3 consultas independentes.
  - Sinal de clarificação quando a pergunta depende de contexto ausente.

- Rastreabilidade:
  - Citações numéricas no corpo da resposta (`[1]`, `[2]`, ...).
  - Lista de fontes montada pelo código com título e URL quando disponível.

- Interface interativa:
  - Chat com histórico curto de memória conversacional.
  - Streaming da resposta.
  - Painel opcional de diagnóstico com consultas e trechos recuperados.

## Arquitetura e Fluxo

1. Ingestão e indexação
- Carrega artigos `.md`.
- Extrai front matter (`title`, `author`, `date`, `url`).
- Limpa ruído recorrente do texto.
- Divide por cabecalhos (`#`, `##`, `###`) e depois por tamanho.
- Gera embeddings (`BAAI/bge-m3`) e persiste no Chroma.

2. Consulta
- Reescreve pergunta para consulta(s) em inglês.
- Executa busca MMR com diversidade.
- Aplica filtro de similaridade para manter apenas trechos relevantes.
- Intercala resultados de múltiplas consultas e remove duplicatas.

3. Geração
- Envia apenas contexto recuperado + histórico curto para o LLM.
- Responde no idioma do usuário.
- Mantem citações numéricas no texto.
- UI anexa bloco de fontes com links quando houver URL.

## Estrutura do Projeto

- `app.py`: interface Streamlit (chat, streaming, detalhes de busca, limpeza de conversa).
- `rag_core.py`: núcleo do pipeline (prompts, reescrita, retrieval, resposta, fontes, setup).
- `query_db.py`: construção do retriever (embeddings, Chroma, MMR, compression filter).
- `ingest.py`: pipeline de ingestão/indexação dos artigos Markdown.
- `chatbot.py`: versao CLI para chat no terminal e depuração rapida.
- `debug_scores.py`: mede scores de similaridade para calibrar threshold.
- `debug_chunks.py`: inspeciona chunks de um artigo e scores para uma query.
- `config.py`: configurações globais (paths, embedding model, device CPU/CUDA).
- `requirements.txt`: dependências principais do projeto.

## Pré-requisitos

- Python 3.10+ (recomendado 3.11).
- Ambiente virtual Python.
- Para backend local:
  - Ollama instalado e em execucao.
  - Modelo baixado: `qwen3:8b`.
- Para backend cloud:
  - Chave da Groq API (`GROQ_API_KEY`).

## Instalação

1. Clonar repositório

```bash
git clone https://github.com/Ali-Maia/seu-repositorio.git
cd seu-repositorio
```

2. Criar e ativar ambiente virtual

Windows (PowerShell):

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Instalar dependências

```bash
pip install -r requirements.txt
```

4. (Opcional) Habilitar backend Groq

```bash
pip install langchain-groq
```

5. Preparar base de conhecimento local (obrigatório)

- Copie os artigos `.md` para o diretório `satisfice_articles/`.
- Esse diretório é ignorado pelo Git e não vem no clone.

## Configuração

Variaveis importantes:

- `LLM_BACKEND`
  - `ollama` (padrao)
  - `groq`
- `GROQ_API_KEY`
  - obrigatória quando `LLM_BACKEND=groq`

Exemplos:

Windows (PowerShell):

```bash
$env:LLM_BACKEND="ollama"
# ou
$env:LLM_BACKEND="groq"
$env:GROQ_API_KEY="sua_chave"
```

Linux/macOS:

```bash
export LLM_BACKEND=ollama
# ou
export LLM_BACKEND=groq
export GROQ_API_KEY="sua_chave"
```

Arquivo `.env` (opcional e recomendado):

```env
LLM_BACKEND=ollama
# Para usar Groq:
# LLM_BACKEND=groq
# GROQ_API_KEY=sua_chave
```

Observação: o arquivo `.env` é local e está no `.gitignore`.

## Itens Não Versionados

Por padrão, os itens abaixo são locais e não são enviados ao repositório:

- `satisfice_articles/` (acervo de artigos)
- `satisfice_articles.zip` (pacote local opcional do acervo)
- `chroma_db_james_bach/` (índice vetorial gerado)
- `.venv/` (ambiente virtual)
- `__pycache__/` (cache Python)
- `.env` (segredos e configuração local)

Se você clonar o projeto em outra máquina, será necessário recriar esses itens localmente.

## Primeira Execução (Fluxo Rápido)

1. Coloque os artigos `.md` em `satisfice_articles/`.
2. Instale as dependências (`pip install -r requirements.txt`).
3. Configure variáveis de ambiente (shell ou `.env`).
4. Gere o índice vetorial (`python ingest.py`).
5. Inicie a interface (`streamlit run app.py`).

## Passo a Passo de Execução

1. Gerar banco vetorial (sempre que atualizar os artigos)

```bash
python ingest.py
```

2. Executar interface web (Streamlit)

```bash
streamlit run app.py
```

3. (Opcional) Usar chatbot no terminal

```bash
python chatbot.py
```

## Como o Sistema Evita Alucinação

- Prompt de resposta com regras estritas de grounding.
- Sem contexto relevante -> resposta explicita de ausência de base.
- Sem complemento com conhecimento externo.
- Citações numéricas obrigatórias para as afirmações suportadas.
- Fontes exibidas separadamente pelo código, não pelo LLM.

## Detalhes Tecnicos Relevantes

- Embeddings normalizados (`normalize_embeddings=True`), permitindo usar produto escalar como aproximação de similaridade coseno.
- MMR configurado para equilíbrio entre relevância e diversidade.
- Filtro de similaridade na compressão de contexto para reduzir ruído.
- Memória de conversa curta (`MAX_TURNS`) para manter foco e custo previsível.
- Streaming de tokens na UI para melhor experiência de uso.

## Scripts de Debug

- Inspecionar score de recuperação para queries variadas:

```bash
python debug_scores.py
```

- Ver chunks de um artigo e score por query:

```bash
python debug_chunks.py
python debug_chunks.py "daily routine of a QA" bharani_asks_about_daily_practice
```

## Solucao de Problemas

- Erro: Ollama não está rodando
  - Inicie o app do Ollama ou rode `ollama serve`.

- Erro: backend `groq` sem pacote
  - Rode `pip install langchain-groq`.

- Erro: `GROQ_API_KEY` ausente
  - Defina a variável de ambiente antes de iniciar o app.

- Reindexação nao atualizou resultados
  - Execute novamente `python ingest.py` para reconstruir o Chroma.

- Falha ao apagar banco durante ingestão
  - Feche processos que usam o Chroma (chat/app em execução) e rode de novo.

<!-- ## Melhorias Futuras (Roadmap)

- Observabilidade do pipeline (latência por etapa, taxa de cobertura de contexto).
- Avaliacao automatica de grounding e qualidade de citacoes.
- Suporte a colecoes multiplas e filtros por tema/ano/autor.
- Persistencia opcional de historico de conversa por usuario.
- Testes automatizados de regressao de retrieval e resposta. -->

## Licença

Este projeto está licenciado sob a licença MIT. Consulte o arquivo `LICENSE` para mais detalhes.

## Créditos

- Acervo fonte: artigos de James Bach em Satisfice.com.
- Implementação: arquitetura RAG com LangChain, Chroma, Hugging Face e Streamlit.
