# -*- coding: utf-8 -*-
"""
Motor de inferência — Arxiv Classifier (Equipe Lorem Ipsum)

Lê um abstract e devolve:
  - a categoria do arXiv em que ele melhor se encaixa (SVM sobre embeddings SPECTER);
  - papers similares do nosso banco de 1.500 artigos (vizinhos por similaridade do cosseno).

O banco vem do arquivo gerado pela Pipeline 3:
  data/arxiv_amostra_1500_com_embeddings.json  (JSON Lines, já com a coluna 'embedding')

Rodar:
  uvicorn main:app --reload --port 8000
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
BASE_DIR   = Path(__file__).parent
DATA_PATH  = Path(os.environ.get("ARXIV_DATA",
                                 BASE_DIR / "data" / "arxiv_amostra_1500_com_embeddings.json"))
MODEL_NAME = os.environ.get("ARXIV_MODEL", "sentence-transformers/allenai-specter")
CACHE_PATH = BASE_DIR / "data" / "svm_cache.joblib"

STATE: dict = {}   # preenchido na inicialização

# ----------------------------------------------------------------------------
# Carregamento do banco + treino do SVM
# ----------------------------------------------------------------------------
def carregar_banco(path: Path):
    """Lê o JSON Lines da Pipeline 3 -> matriz de embeddings + metadados."""
    embeddings, meta = [], []
    with open(path, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            r = json.loads(linha)
            embeddings.append(r["embedding"])
            meta.append({
                "id": str(r["id"]),
                "title": (r.get("title") or "").strip(),
                "category": r["assigned_category"],
                "abstract": (r.get("abstract_reduzido") or r.get("abstract") or "").strip(),
            })
    X = np.asarray(embeddings, dtype=np.float32)
    return X, meta


def treinar_svm(X, y):
    """SVM linear (com padronização) + probabilidades calibradas para a confiança."""
    from sklearn.calibration import CalibratedClassifierCV
    clf = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(kernel="linear", random_state=42), ensemble=False),
    )
    clf.fit(X, y)
    return clf


def fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{path.name}-{st.st_size}-{int(st.st_mtime)}"


def carregar_modelo_svm(X, y, path: Path):
    """Treina o SVM uma vez e guarda em cache (joblib) p/ reinícios instantâneos."""
    import joblib
    fp = fingerprint(path)
    if CACHE_PATH.exists():
        try:
            cache = joblib.load(CACHE_PATH)
            if cache.get("fp") == fp:
                return cache["clf"]
        except Exception:
            pass
    clf = treinar_svm(X, y)
    try:
        joblib.dump({"fp": fp, "clf": clf}, CACHE_PATH)
    except Exception:
        pass
    return clf


# ----------------------------------------------------------------------------
# Ciclo de vida — carrega tudo quando o servidor sobe
# ----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DATA_PATH.exists():
        raise RuntimeError(
            f"Banco não encontrado em {DATA_PATH}. "
            "Copie 'arxiv_amostra_1500_com_embeddings.json' (saída da Pipeline 3) para a pasta backend/data/."
        )
    print(f"[inferencia] carregando banco de {DATA_PATH} ...")
    X, meta = carregar_banco(DATA_PATH)
    y = np.array([m["category"] for m in meta], dtype=object)

    print(f"[inferencia] treinando SVM em {X.shape[0]} artigos x {X.shape[1]} dims ...")
    clf = carregar_modelo_svm(X, y, DATA_PATH)

    print(f"[inferencia] carregando modelo de embeddings '{MODEL_NAME}' (pode baixar na 1a vez) ...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(MODEL_NAME)

    STATE.update(
        X=X, meta=meta, y=y, clf=clf,
        classes=clf.classes_, encoder=encoder,
        sep=encoder.tokenizer.sep_token or "[SEP]",
    )
    print(f"[inferencia] pronto. {len(meta)} papers, {len(clf.classes_)} categorias.")
    yield
    STATE.clear()


app = FastAPI(title="Arxiv Classifier — motor de inferência", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # dev: libera o front (Vite em localhost:5173)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------------
class ClassifyIn(BaseModel):
    abstract: str
    title: str | None = ""
    top_k: int = 5


class CatScore(BaseModel):
    category: str
    score: float


class SimilarPaper(BaseModel):
    id: str
    title: str
    category: str
    similarity: float
    url: str


class ClassifyOut(BaseModel):
    predicted_category: str
    confidence: float
    ranking: list[CatScore]
    similar_papers: list[SimilarPaper]


# ----------------------------------------------------------------------------
# Rotas
# ----------------------------------------------------------------------------
@app.get("/")
def health():
    return {
        "status": "ok" if STATE else "carregando",
        "papers": len(STATE.get("meta", [])),
        "categories": sorted(map(str, STATE.get("classes", []))),
        "model": MODEL_NAME,
    }


@app.post("/classify", response_model=ClassifyOut)
def classify(req: ClassifyIn):
    if not req.abstract or not req.abstract.strip():
        raise HTTPException(status_code=400, detail="O campo 'abstract' está vazio.")

    encoder, clf = STATE["encoder"], STATE["clf"]
    sep, classes, X, meta = STATE["sep"], STATE["classes"], STATE["X"], STATE["meta"]

    # mesma entrada da Pipeline 3: "título [SEP] abstract"
    titulo = (req.title or "").strip()
    texto = f"{titulo} {sep} {req.abstract.strip()}" if titulo else req.abstract.strip()
    q = encoder.encode([texto], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)

    # classificação (probabilidades -> ranking + confiança)
    proba = clf.predict_proba(q.reshape(1, -1))[0]
    ordem = np.argsort(proba)[::-1]
    n_rank = max(1, min(req.top_k, len(classes)))
    ranking = [CatScore(category=str(classes[i]), score=float(proba[i])) for i in ordem[:n_rank]]

    # papers similares (cosseno = produto interno, tudo normalizado)
    sims = X @ q
    k = max(1, min(req.top_k, 20))
    top = np.argsort(sims)[::-1][:k]
    similares = [
        SimilarPaper(
            id=meta[i]["id"],
            title=meta[i]["title"],
            category=meta[i]["category"],
            similarity=float(sims[i]),
            url=f"https://arxiv.org/abs/{meta[i]['id']}",
        )
        for i in top
    ]

    return ClassifyOut(
        predicted_category=ranking[0].category,
        confidence=ranking[0].score,
        ranking=ranking,
        similar_papers=similares,
    )
