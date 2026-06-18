# projeto_ia — Arxiv Classifier

Projeto de Inteligência Artificial da Universidade Politécnica de Pernambuco — Equipe Lorem Ipsum.

A aplicação lê o **abstract** de um artigo científico e devolve:
- a **categoria do arXiv** em que ele melhor se encaixa;
- **papers similares** do nosso banco de 1.500 artigos.

## Arquitetura

```
Front-end (React + Vite)  ──HTTP──>  Back-end (FastAPI)
  cola/busca o abstract                embedding SPECTER -> SVM (categoria)
  exibe categoria + similares          + vizinhos por cosseno (similares)
```

O modelo (SPECTER + SVM) roda no back-end em Python, porque não é viável rodar no navegador.

## Como rodar (duas partes)

### 1. Back-end (motor de inferência)

```bash
cd backend
# copie a saída da Pipeline 3 para backend/data/:
#   data/arxiv_amostra_1500_com_embeddings.json
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Detalhes em [`backend/README.md`](backend/README.md). A 1ª execução baixa o modelo SPECTER (~440 MB).

### 2. Front-end

```bash
npm install
npm run dev          # abre em http://localhost:5173
```

O front fala com o back-end em `http://localhost:8000` por padrão. Para apontar para outro
endereço, crie um arquivo `.env` na raiz com:

```
VITE_API_URL=http://meu-host:8000
```

## Fluxo de uso

1. Cole um abstract (ou busque pelo link do arXiv, que também traz o título).
2. Clique em **Classificar abstract**.
3. O painel mostra a categoria prevista (com confiança), outras categorias prováveis e os papers similares (com link para o arXiv).
