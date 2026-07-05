# Motor de inferência — Arxiv Classifier

Backend FastAPI que recebe um abstract e devolve a categoria primaria, categorias
multi-label aplicaveis e papers similares. Quando os artefatos 40k ainda nao existem,
o servico usa automaticamente o classificador single-label de 10.500 artigos.

## 1. Dados

Copie a saída da **Pipeline 3** para a pasta `data/`:

```
backend/data/arxiv_amostra_10500_com_embeddings_atualizada.json
```

Para ativar o modo calibrado multi-label de 40.000 artigos, copie tambem as duas
saidas das novas Pipelines 3 e 4:

```text
backend/data/arxiv_amostra_40000_com_embeddings_multilabel.json
backend/data/arxiv_modelos_40000_calibrados_multilabel.joblib
```

Ao reiniciar o backend, `GET /` deve informar
`"inference_mode": "calibrated_multilabel_40000"`.

Esse arquivo (JSON Lines) já contém os embeddings + título + abstract + categoria de cada paper.

## 2. Instalar e rodar

```bash
cd backend
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Recomendado: instale o PyTorch CPU-only ANTES (evita baixar ~2,5-5 GB de CUDA da NVIDIA).
# A inferência aqui roda tranquila na CPU, então não precisamos da versão para GPU.
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

> **Tamanho das dependências:** o `pip install torch` padrão no Linux puxa as bibliotecas CUDA
> e ocupa vários GB. Instalando o torch CPU-only (comando acima) o ambiente fica em torno de
> ~1,2 GB. Como o torch já estará instalado, o `pip install -r requirements.txt` seguinte só
> baixa o resto (leve).
>
> Na **primeira execução** o modelo SPECTER (~440 MB) é baixado automaticamente e fica em cache
> (em `~/.cache`). O SVM é treinado na inicialização (~segundos) e guardado em `data/svm_cache.joblib`.

## 3. Endpoints

- `GET  /`         — status do serviço (nº de papers, categorias, modelo).
- `POST /classify` — classifica um abstract.

Exemplo:

```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"abstract": "We propose a deep neural network for image classification...", "top_k": 3, "similar_k": 5}'
```

Resposta:

```json
{
  "predicted_category": "cs.CV",
  "confidence": 0.82,
  "ranking": [{"category": "cs.CV", "score": 0.82}, {"category": "cs.LG", "score": 0.11}],
  "similar_papers": [
    {"id": "1234.5678", "title": "...", "category": "cs.CV", "similarity": 0.79,
     "url": "https://arxiv.org/abs/1234.5678"}
  ]
}
```

## 4. Configuração (variáveis de ambiente, opcionais)

- `ARXIV_DATA`  — caminho do JSON do banco (padrão: `data/arxiv_amostra_10500_com_embeddings_atualizada.json`).
- `ARXIV_MODEL` — modelo de embeddings (padrão: `sentence-transformers/allenai-specter`).
- `ARXIV_SVM_C` — hiperparâmetro C do SVM linear (padrão: `0.03`).
- `ARXIV_MODEL_BUNDLE` — caminho do bundle calibrado multi-label gerado pela Pipeline 4 de 40k.

O front-end (React/Vite) consome este backend; veja a variável `VITE_API_URL` no projeto raiz.
