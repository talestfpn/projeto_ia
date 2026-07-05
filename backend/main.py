# -*- coding: utf-8 -*-
"""
Motor de inferência — Arxiv Classifier (Equipe Lorem Ipsum)

Lê um abstract e devolve:
  - a categoria do arXiv em que ele melhor se encaixa (SVM sobre embeddings SPECTER);
  - papers similares do nosso banco de artigos (vizinhos por similaridade do cosseno).

O banco vem do arquivo gerado pela Pipeline 3:
  data/arxiv_amostra_10500_com_embeddings_atualizada.json  (JSON Lines, já com a coluna 'embedding')

Rodar:
  uvicorn main:app --reload --port 8000
"""
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from io import BytesIO
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
DATA_40K_PATH = BASE_DIR / "data" / "arxiv_amostra_40000_com_embeddings_multilabel.json"
MODEL_BUNDLE_PATH = Path(os.environ.get(
    "ARXIV_MODEL_BUNDLE",
    BASE_DIR / "data" / "arxiv_modelos_40000_calibrados_multilabel.joblib",
))
LEGACY_DATA_PATH = BASE_DIR / "data" / "arxiv_amostra_10500_com_embeddings_atualizada.json"
DEFAULT_DATA_PATH = DATA_40K_PATH if DATA_40K_PATH.exists() and MODEL_BUNDLE_PATH.exists() else LEGACY_DATA_PATH
DATA_PATH = Path(os.environ.get("ARXIV_DATA", DEFAULT_DATA_PATH))
MODEL_NAME = os.environ.get("ARXIV_MODEL", "sentence-transformers/allenai-specter")
SVM_C = float(os.environ.get("ARXIV_SVM_C", "0.03"))
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
            primary_category = r.get("primary_category") or r.get("assigned_category")
            if not primary_category:
                raise ValueError(f"Artigo {r.get('id')} sem categoria primaria.")
            target_categories = r.get("target_categories")
            if not target_categories:
                target_categories = [primary_category]
            meta.append({
                "id": str(r["id"]),
                "title": (r.get("title") or "").strip(),
                "category": primary_category,
                "categories": list(target_categories),
                "abstract": (r.get("abstract_reduzido") or r.get("abstract") or "").strip(),
            })
    X = np.asarray(embeddings, dtype=np.float32)
    return X, meta


def treinar_svm(X, y):
    """SVM linear (com padronização) + probabilidades calibradas para a confiança."""
    from sklearn.calibration import CalibratedClassifierCV
    clf = make_pipeline(
        StandardScaler(),
        CalibratedClassifierCV(SVC(kernel="linear", C=SVM_C, random_state=42), ensemble=False),
    )
    clf.fit(X, y)
    return clf


def fingerprint(path: Path) -> str:
    st = path.stat()
    return f"{path.name}-{st.st_size}-{int(st.st_mtime)}-svc-c-{SVM_C:g}"


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
            "Copie 'arxiv_amostra_10500_com_embeddings_atualizada.json' (saída da Pipeline 3) para a pasta backend/data/."
        )
    print(f"[inferencia] carregando banco de {DATA_PATH} ...")
    X, meta = carregar_banco(DATA_PATH)
    y = np.array([m["category"] for m in meta], dtype=object)

    dual_head = MODEL_BUNDLE_PATH.exists() and DATA_PATH.name != LEGACY_DATA_PATH.name
    if dual_head:
        import joblib

        print(f"[inferencia] carregando modelos calibrados de {MODEL_BUNDLE_PATH} ...")
        bundle = joblib.load(MODEL_BUNDLE_PATH)
        required = {
            "primary_model", "multilabel_model", "primary_classes",
            "multilabel_classes", "thresholds",
        }
        missing = required.difference(bundle)
        if missing:
            raise RuntimeError(f"Bundle 40k incompleto. Campos ausentes: {sorted(missing)}")
        inference_mode = "calibrated_multilabel_40000"
        clf = None
    else:
        print(f"[inferencia] treinando SVM legado em {X.shape[0]} artigos x {X.shape[1]} dims (C={SVM_C:g}) ...")
        clf = carregar_modelo_svm(X, y, DATA_PATH)
        bundle = None
        inference_mode = "legacy_single_label"

    print(f"[inferencia] carregando modelo de embeddings '{MODEL_NAME}' (pode baixar na 1a vez) ...")
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer(MODEL_NAME)

    classes = bundle["primary_classes"] if bundle else list(clf.classes_)
    STATE.update(
        X=X, meta=meta, y=y, clf=clf, bundle=bundle,
        classes=np.asarray(classes, dtype=object), encoder=encoder,
        sep=encoder.tokenizer.sep_token or "[SEP]", inference_mode=inference_mode,
    )
    print(f"[inferencia] pronto. {len(meta)} papers, {len(classes)} categorias, modo={inference_mode}.")
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
    top_k: int = 3
    similar_k: int = 5


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
    applicable_categories: list[CatScore]
    inference_mode: str
    similar_papers: list[SimilarPaper]


class ArxivPaperOut(BaseModel):
    id: str
    title: str
    abstract: str


class PdfAbstractOut(BaseModel):
    filename: str
    title: str
    abstract: str


def extrair_abstract_pdf(texto: str) -> tuple[str, str]:
    linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]
    texto_com_quebras = "\n".join(linhas)
    match = re.search(
        r"\babstract\b\s*[:.\-]?\s*(.+?)(?=\n\s*(?:1\.?\s*)?(?:introduction|keywords|index terms)\b|\n\s*i\.\s*introduction\b)",
        texto_com_quebras,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match:
        abstract = match.group(1)
    else:
        fallback = re.search(r"\babstract\b\s*[:.\-]?\s*(.+)", texto_com_quebras, flags=re.IGNORECASE | re.DOTALL)
        if fallback:
            abstract = fallback.group(1)[:2200]
        else:
            # Alguns PDFs do arXiv/html não imprimem o rótulo "Abstract"; nesses casos
            # o resumo costuma ser o primeiro bloco longo antes de links/rodapé/introdução.
            texto_sem_rodape = re.split(
                r"\b(?:project website|code|keywords|index terms|1\.?\s*introduction|i\.\s*introduction|introduction)\b",
                texto_com_quebras,
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0]
            linhas_uteis = [
                linha for linha in texto_sem_rodape.splitlines()
                if not re.search(r"https?://|@|university|institute|laboratory|arxiv:|^\[|\b\d{1,2}\s+\w+\s+20\d{2}\b", linha, re.I)
            ]
            candidatos: list[str] = []
            for i, linha in enumerate(linhas_uteis):
                inicio_parece_resumo = (
                    len(linha) >= 45
                    and re.search(r"\b(?:we|this paper|this work|we propose|we introduce|we present|provides|however|in this)\b", linha, re.I)
                    and not re.search(r"\b(?:abstract|author|department|proceedings)\b", linha, re.I)
                )
                if not inicio_parece_resumo:
                    continue

                bloco = " ".join(linhas_uteis[i:i + 14])
                bloco = re.sub(r"\s+", " ", bloco).strip()
                if len(bloco) >= 220 and len(bloco.split()) >= 35:
                    candidatos.append(bloco[:2200])
                    break

            if not candidatos:
                raise HTTPException(
                    status_code=422,
                    detail="Não foi possível localizar a seção Abstract no PDF.",
                )
            abstract = candidatos[0]

    abstract = re.sub(r"\s+", " ", abstract).strip()
    if len(abstract) < 80:
        raise HTTPException(status_code=422, detail="Abstract extraído ficou curto demais.")

    titulo = ""
    abstract_line = next((i for i, linha in enumerate(linhas) if re.search(r"\babstract\b", linha, re.I)), -1)
    if abstract_line > 0:
        candidatos = [linha for linha in linhas[max(0, abstract_line - 8):abstract_line] if 12 <= len(linha) <= 220]
        titulo = candidatos[0] if candidatos else ""

    return titulo, abstract


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
        "data_file": DATA_PATH.name,
        "model_bundle": MODEL_BUNDLE_PATH.name if MODEL_BUNDLE_PATH.exists() else None,
        "inference_mode": STATE.get("inference_mode", "carregando"),
        "svm_c": SVM_C,
    }


@app.get("/arxiv/{paper_id}", response_model=ArxivPaperOut)
def fetch_arxiv_paper(paper_id: str):
    safe_id = paper_id.strip()
    if not safe_id:
        raise HTTPException(status_code=400, detail="ID do arXiv vazio.")

    query = urllib.parse.urlencode({"id_list": safe_id})
    url = f"https://export.arxiv.org/api/query?{query}"

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            xml_body = response.read()
    except urllib.error.URLError as exc:
        raise HTTPException(status_code=502, detail="Erro ao consultar a API do arXiv.") from exc

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(xml_body)
    entry = root.find("atom:entry", ns)
    if entry is None:
        raise HTTPException(status_code=404, detail="Artigo não encontrado no arXiv.")

    title = entry.findtext("atom:title", default="", namespaces=ns)
    summary = entry.findtext("atom:summary", default="", namespaces=ns)
    arxiv_id = entry.findtext("atom:id", default="", namespaces=ns).rstrip("/").split("/")[-1]

    abstract = " ".join(summary.split())
    if not abstract:
        raise HTTPException(status_code=404, detail="Abstract não encontrado no arXiv.")

    return ArxivPaperOut(
        id=arxiv_id or safe_id,
        title=" ".join(title.split()),
        abstract=abstract,
    )


@app.post("/pdf/abstract", response_model=PdfAbstractOut)
async def extract_pdf_abstract(file: UploadFile = File(...)):
    filename = file.filename or "paper.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Envie um arquivo PDF.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="PDF vazio.")

    try:
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        paginas = reader.pages[: min(5, len(reader.pages))]
        texto = "\n".join(page.extract_text() or "" for page in paginas)
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Não foi possível ler o texto do PDF.") from exc

    if not texto.strip():
        raise HTTPException(status_code=422, detail="O PDF não possui texto extraível.")

    title, abstract = extrair_abstract_pdf(texto)
    return PdfAbstractOut(filename=filename, title=title, abstract=abstract)


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

    # Categoria primaria ordenada e categorias multi-label independentes.
    bundle = STATE.get("bundle")
    if bundle:
        primary_proba = bundle["primary_model"].predict_proba(q.reshape(1, -1))[0]
        primary_classes = np.asarray(bundle["primary_classes"], dtype=object)
        ordem = np.argsort(primary_proba)[::-1]
        n_rank = max(1, min(req.top_k, len(primary_classes)))
        ranking = [
            CatScore(category=str(primary_classes[i]), score=float(primary_proba[i]))
            for i in ordem[:n_rank]
        ]

        multi_proba = bundle["multilabel_model"].predict_proba(q.reshape(1, -1))[0]
        multi_classes = list(map(str, bundle["multilabel_classes"]))
        thresholds = bundle["thresholds"]
        selected = {
            category: float(score)
            for category, score in zip(multi_classes, multi_proba)
            if float(score) >= float(thresholds[category])
        }
        primary_multi_idx = multi_classes.index(ranking[0].category)
        selected[ranking[0].category] = float(multi_proba[primary_multi_idx])
        applicable = [
            CatScore(category=category, score=score)
            for category, score in sorted(selected.items(), key=lambda item: item[1], reverse=True)
        ]
    else:
        proba = clf.predict_proba(q.reshape(1, -1))[0]
        ordem = np.argsort(proba)[::-1]
        n_rank = max(1, min(req.top_k, len(classes)))
        ranking = [CatScore(category=str(classes[i]), score=float(proba[i])) for i in ordem[:n_rank]]
        applicable = [ranking[0]]

    # papers similares (cosseno = produto interno, tudo normalizado)
    sims = X @ q
    k = max(1, min(req.similar_k, 20))
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
        applicable_categories=applicable,
        inference_mode=STATE["inference_mode"],
        similar_papers=similares,
    )
