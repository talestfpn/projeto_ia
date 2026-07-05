import { ChangeEvent, FormEvent, useMemo, useState } from 'react';

type Language = 'pt' | 'en';
type ArxivStatus = 'idle' | 'loading' | 'success' | 'error';
type PdfStatus = 'idle' | 'loading' | 'success' | 'error';
type AnalysisStatus = 'idle' | 'loading' | 'success' | 'error';

const API_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? 'http://localhost:8000';

interface CatScore {
  category: string;
  score: number;
}

interface SimilarPaper {
  id: string;
  title: string;
  category: string;
  similarity: number;
  url: string;
}

interface ClassifyResult {
  predicted_category: string;
  confidence: number;
  ranking: CatScore[];
  similar_papers: SimilarPaper[];
}

interface ArxivPaper {
  id: string;
  title: string;
  abstract: string;
}

interface PdfAbstract {
  filename: string;
  title: string;
  abstract: string;
}

const copy = {
  pt: {
    eyebrow: 'Classificador de artigos',
    title: 'Arxiv Classifier',
    subtitle:
      'Cole o abstract de um artigo científico e descubra em qual categoria do arXiv ele melhor se encaixa, além de papers similares do nosso banco.',
    language: 'Idioma',
    arxivTitle: 'Link do arXiv',
    arxivHint:
      'Cole o link do artigo no arXiv para buscar o abstract automaticamente.',
    arxivPlaceholder: 'Ex.: https://arxiv.org/abs/2301.12345',
    arxivButton: 'Buscar abstract',
    arxivLoading: 'Buscando...',
    arxivSuccess: 'Abstract carregado com sucesso',
    arxivInvalidUrl: 'URL inválida. Use o formato https://arxiv.org/abs/XXXX.XXXXX',
    arxivFetchError: 'Erro ao buscar o artigo. Verifique o link e tente novamente.',
    uploadTitle: 'Anexar paper em PDF',
    uploadHint: 'Selecione um PDF para extrair automaticamente o abstract.',
    uploadButton: 'Escolher PDF',
    noFile: 'Nenhum arquivo selecionado',
    fileReady: 'Arquivo pronto para demonstração',
    pdfLoading: 'Extraindo abstract do PDF...',
    pdfSuccess: 'Abstract extraído do PDF',
    pdfExtractError:
      'Não foi possível extrair o abstract desse PDF. Cole o resumo manualmente.',
    abstractTitle: 'Colar abstract',
    abstractHint:
      'Cole o resumo do artigo que será enviado ao motor de classificação.',
    abstractPlaceholder:
      'Ex.: We propose a neural approach for representation learning in scientific documents...',
    chars: 'caracteres',
    analyze: 'Classificar abstract',
    analyzing: 'Classificando...',
    validation: 'Cole um abstract (ou busque pelo link do arXiv) antes de classificar.',
    backendError:
      'Não foi possível conectar ao motor de inferência. Confira se o backend está rodando em ',
    resultTitle: 'Retorno do sistema',
    idleResult:
      'A área de retorno está pronta. Cole um abstract e clique em "Classificar abstract" para ver a categoria prevista e papers similares.',
    predictedTitle: 'Categoria prevista',
    confidenceLabel: 'confiança',
    topCategories: 'Top 3 categorias',
    topCategoryHint: 'Ranking das categorias mais prováveis para este abstract.',
    topPrediction: 'Top 1',
    similarTitle: 'Papers similares',
    similarHint: 'Os mais próximos no banco de 10.500 artigos.',
    similarityLabel: 'similaridade',
    pipelineTitle: 'Pipeline',
    pipeline: ['Entrada do abstract', 'Embedding (SPECTER)', 'SVM + vizinhos', 'Categoria + similares'],
    datasetTitle: 'Base de conhecimento',
    datasetText:
      'O projeto usa o arXiv Dataset da Cornell University como fonte de abstracts, títulos, autores e categorias. Nesta demo local, o banco usa 10.500 artigos com embeddings.',
    categoriesTitle: 'Categorias',
    categoriesText:
      'O sistema classifica nas 15 maiores categorias finas do projeto (cs.LG, hep-ph, cs.CV, astro-ph, e outras).',
    singleTitle: 'Embeddings SPECTER',
    singleText:
      'Cada abstract vira um vetor de 768 dimensões com um BERT treinado em artigos científicos.',
    multiTitle: 'Papers similares',
    multiText:
      'A similaridade do cosseno entre embeddings recupera os artigos mais parecidos do banco.',
    modelTitle: 'Classificador',
    modelText:
      'Um SVM linear treinado sobre os embeddings decide o Top 3 de categorias mais prováveis do abstract.',
    sampleTags: ['cs.LG', 'hep-ph', 'cs.CV', 'astro-ph', 'quant-ph'],
  },
  en: {
    eyebrow: 'Paper classifier',
    title: 'Arxiv Classifier',
    subtitle:
      'Paste a scientific paper abstract and discover which arXiv category it best fits, plus similar papers from our database.',
    language: 'Language',
    arxivTitle: 'arXiv link',
    arxivHint: 'Paste an arXiv article URL to automatically fetch its abstract.',
    arxivPlaceholder: 'E.g.: https://arxiv.org/abs/2301.12345',
    arxivButton: 'Fetch abstract',
    arxivLoading: 'Fetching...',
    arxivSuccess: 'Abstract loaded successfully',
    arxivInvalidUrl: 'Invalid URL. Use the format https://arxiv.org/abs/XXXX.XXXXX',
    arxivFetchError: 'Failed to fetch the paper. Check the link and try again.',
    uploadTitle: 'Attach paper PDF',
    uploadHint: 'Select a PDF to automatically extract its abstract.',
    uploadButton: 'Choose PDF',
    noFile: 'No file selected',
    fileReady: 'File ready for demo',
    pdfLoading: 'Extracting abstract from PDF...',
    pdfSuccess: 'Abstract extracted from PDF',
    pdfExtractError: 'Could not extract the abstract from this PDF. Paste it manually.',
    abstractTitle: 'Paste abstract',
    abstractHint:
      'Paste the paper abstract that will be sent to the classification engine.',
    abstractPlaceholder:
      'Example: We propose a neural approach for representation learning in scientific documents...',
    chars: 'characters',
    analyze: 'Classify abstract',
    analyzing: 'Classifying...',
    validation: 'Paste an abstract (or fetch it from an arXiv link) before classifying.',
    backendError:
      'Could not reach the inference engine. Make sure the backend is running at ',
    resultTitle: 'System output',
    idleResult:
      'The output area is ready. Paste an abstract and click "Classify abstract" to see the predicted category and similar papers.',
    predictedTitle: 'Predicted category',
    confidenceLabel: 'confidence',
    topCategories: 'Top 3 categories',
    topCategoryHint: 'Ranking of the most likely categories for this abstract.',
    topPrediction: 'Top 1',
    similarTitle: 'Similar papers',
    similarHint: 'The closest ones in the 10,500-paper database.',
    similarityLabel: 'similarity',
    pipelineTitle: 'Pipeline',
    pipeline: ['Abstract input', 'Embedding (SPECTER)', 'SVM + neighbors', 'Category + similar'],
    datasetTitle: 'Knowledge base',
    datasetText:
      'The project uses the Cornell University arXiv Dataset as a source of abstracts, titles, authors, and categories. This local demo uses 10,500 papers with embeddings.',
    categoriesTitle: 'Categories',
    categoriesText:
      'The system classifies into the 15 largest fine-grained categories (cs.LG, hep-ph, cs.CV, astro-ph, and others).',
    singleTitle: 'SPECTER embeddings',
    singleText:
      'Each abstract becomes a 768-dimensional vector using a BERT trained on scientific papers.',
    multiTitle: 'Similar papers',
    multiText:
      'Cosine similarity between embeddings retrieves the most alike articles in the database.',
    modelTitle: 'Classifier',
    modelText:
      'A linear SVM trained on the embeddings returns the Top 3 most likely categories for the abstract.',
    sampleTags: ['cs.LG', 'hep-ph', 'cs.CV', 'astro-ph', 'quant-ph'],
  },
} satisfies Record<Language, Record<string, string | string[]>>;

function parseArxivId(input: string): string | null {
  const trimmed = input.trim();
  const match = trimmed.match(/(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:\.[A-Z]+)?\/\d{7}(?:v\d+)?)/);
  if (!match) return null;
  return match[1].replace(/v\d+$/, '');
}

function App() {
  const [language, setLanguage] = useState<Language>('pt');
  const [fileName, setFileName] = useState('');
  const [pdfStatus, setPdfStatus] = useState<PdfStatus>('idle');
  const [pdfMessage, setPdfMessage] = useState('');
  const [abstractText, setAbstractText] = useState('');
  const [arxivUrl, setArxivUrl] = useState('');
  const [arxivStatus, setArxivStatus] = useState<ArxivStatus>('idle');
  const [arxivMessage, setArxivMessage] = useState('');
  const [paperTitle, setPaperTitle] = useState('');
  const [analysis, setAnalysis] = useState<ClassifyResult | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState<AnalysisStatus>('idle');
  const [analysisError, setAnalysisError] = useState('');

  const t = copy[language];
  const abstractCount = useMemo(() => abstractText.trim().length, [abstractText]);

  function resetAnalysis() {
    setAnalysis(null);
    setAnalysisStatus('idle');
    setAnalysisError('');
  }

  async function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    setFileName(file?.name ?? '');
    resetAnalysis();
    if (!file) {
      setPdfStatus('idle');
      setPdfMessage('');
      return;
    }

    setPdfStatus('loading');
    setPdfMessage(String(t.pdfLoading));

    const body = new FormData();
    body.append('file', file);

    try {
      const res = await fetch(`${API_URL}/pdf/abstract`, {
        method: 'POST',
        body,
      });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data: PdfAbstract = await res.json();
      setAbstractText(data.abstract);
      setPaperTitle(data.title);
      setPdfStatus('success');
      setPdfMessage(String(t.pdfSuccess));
    } catch {
      setPdfStatus('error');
      setPdfMessage(String(t.pdfExtractError));
    }
  }

  function handleAbstractChange(event: ChangeEvent<HTMLTextAreaElement>) {
    setAbstractText(event.target.value);
    setPaperTitle(''); // abstract digitado/colado manualmente: sem título associado
    resetAnalysis();
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!abstractText.trim()) {
      setAnalysis(null);
      setAnalysisStatus('error');
      setAnalysisError(String(t.validation));
      return;
    }
    setAnalysisStatus('loading');
    setAnalysisError('');
    setAnalysis(null);
    try {
      const res = await fetch(`${API_URL}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          abstract: abstractText.trim(),
          title: paperTitle.trim(),
          top_k: 3,
          similar_k: 5,
        }),
      });
      if (!res.ok) throw new Error(`status ${res.status}`);
      const data: ClassifyResult = await res.json();
      setAnalysis(data);
      setAnalysisStatus('success');
    } catch {
      setAnalysisStatus('error');
      setAnalysisError(`${t.backendError}${API_URL}.`);
    }
  }

  async function handleFetchArxiv() {
    const id = parseArxivId(arxivUrl);
    if (!id) {
      setArxivStatus('error');
      setArxivMessage(String(t.arxivInvalidUrl));
      return;
    }
    setArxivStatus('loading');
    setArxivMessage('');
    try {
      const res = await fetch(`${API_URL}/arxiv/${encodeURIComponent(id)}`);
      if (!res.ok) throw new Error(`status ${res.status}`);
      const paper: ArxivPaper = await res.json();
      setAbstractText(paper.abstract);
      setPaperTitle(paper.title);
      setArxivStatus('success');
      setArxivMessage(String(t.arxivSuccess));
      resetAnalysis();
    } catch {
      setArxivStatus('error');
      setArxivMessage(String(t.arxivFetchError));
    }
  }

  return (
    <main className="app-shell">
      <div className="background-grid" aria-hidden="true" />
      <section className="hero">
        <nav className="topbar glass">
          <div className="brand-mark" aria-label="Arxiv Classifier">
            <span>Ax</span>
          </div>
          <div className="language-switch" aria-label={String(t.language)}>
            <span>{t.language}</span>
            <button
              className={language === 'pt' ? 'active' : ''}
              type="button"
              onClick={() => setLanguage('pt')}
            >
              PT
            </button>
            <button
              className={language === 'en' ? 'active' : ''}
              type="button"
              onClick={() => setLanguage('en')}
            >
              EN
            </button>
          </div>
        </nav>

        <div className="hero-content">
          <div className="intro">
            <p className="eyebrow">{t.eyebrow}</p>
            <h1>{t.title}</h1>
            <p className="subtitle">{t.subtitle}</p>
            <div className="tag-row" aria-label="arXiv categories">
              {(t.sampleTags as string[]).map((tag) => (
                <span key={tag}>{tag}</span>
              ))}
            </div>
          </div>

          <aside className="paper-preview glass" aria-label="Scientific paper preview">
            <div className="paper-header">
              <span />
              <span />
              <span />
            </div>
            <div className="paper-line wide" />
            <div className="paper-line" />
            <div className="paper-line short" />
            <div className="paper-equation">{'Abstract -> SPECTER -> SVM | cosseno'}</div>
            <div className="paper-tags">
              <span>cs.LG</span>
              <span>math.MP</span>
              <span>astro-ph</span>
            </div>
          </aside>
        </div>
      </section>

      <section className="workspace">
        <form className="input-panel glass" onSubmit={handleSubmit}>
          <div className="section-heading">
            <span className="section-index">01</span>
            <div>
              <h2>{t.arxivTitle}</h2>
              <p>{t.arxivHint}</p>
            </div>
          </div>

          <div className="arxiv-input-row">
            <input
              className="arxiv-url-input"
              type="url"
              value={arxivUrl}
              onChange={(e) => {
                setArxivUrl(e.target.value);
                setArxivStatus('idle');
                setArxivMessage('');
              }}
              placeholder={String(t.arxivPlaceholder)}
            />
            <button
              type="button"
              className="arxiv-fetch-btn"
              onClick={handleFetchArxiv}
              disabled={arxivStatus === 'loading'}
            >
              {arxivStatus === 'loading' ? t.arxivLoading : t.arxivButton}
            </button>
          </div>

          {arxivMessage ? (
            <div className={`status-pill ${arxivStatus === 'success' ? 'success' : 'error'}`}>
              {arxivMessage}
            </div>
          ) : null}

          <div className="section-heading compact">
            <span className="section-index">02</span>
            <div>
              <h2>{t.uploadTitle}</h2>
              <p>{t.uploadHint}</p>
            </div>
          </div>

          <label className="upload-zone">
            <input accept="application/pdf,.pdf" type="file" onChange={handleFileChange} />
            <span className="upload-icon" aria-hidden="true">
              PDF
            </span>
            <strong>{t.uploadButton}</strong>
            <small>{fileName || t.noFile}</small>
          </label>

          {pdfMessage ? (
            <div className={`status-pill ${pdfStatus === 'success' ? 'success' : pdfStatus === 'error' ? 'error' : 'neutral'}`}>
              {pdfMessage}
            </div>
          ) : null}

          <div className="section-heading compact">
            <span className="section-index">03</span>
            <div>
              <h2>{t.abstractTitle}</h2>
              <p>{t.abstractHint}</p>
            </div>
          </div>

          <textarea
            value={abstractText}
            onChange={handleAbstractChange}
            placeholder={String(t.abstractPlaceholder)}
          />
          <div className="input-footer">
            <span>
              {abstractCount} {t.chars}
            </span>
            <button type="submit" disabled={analysisStatus === 'loading'}>
              {analysisStatus === 'loading' ? t.analyzing : t.analyze}
            </button>
          </div>
        </form>

        <section className="result-panel glass" aria-live="polite">
          <div className="result-header">
            <span className="section-index">04</span>
            <h2>{t.resultTitle}</h2>
          </div>

          {analysisStatus === 'error' ? (
            <div className="message warning">{analysisError}</div>
          ) : null}

          {analysisStatus === 'loading' ? (
            <div className="development-state">
              <span className="status-pill neutral">SPECTER + SVM</span>
              <h3>{t.analyzing}</h3>
            </div>
          ) : null}

          {analysisStatus === 'success' && analysis ? (
            <div className="analysis-result">
              <div className="predicted-card">
                <span className="predicted-label">{t.predictedTitle}</span>
                <strong className="predicted-category">{analysis.predicted_category}</strong>
                <span className="prediction-rank">{t.topPrediction}</span>
                <div className="confidence-row">
                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{ width: `${Math.round(analysis.confidence * 100)}%` }}
                    />
                  </div>
                  <span className="confidence-value">
                    {Math.round(analysis.confidence * 100)}% {t.confidenceLabel}
                  </span>
                </div>
              </div>

              {analysis.ranking.length > 0 ? (
                <div className="ranking-block">
                  <h3>{t.topCategories}</h3>
                  <p className="ranking-hint">{t.topCategoryHint}</p>
                  <div className="ranking-list">
                    {analysis.ranking.slice(0, 3).map((c, index) => (
                      <div className="ranking-item" key={c.category}>
                        <span className="ranking-position">{index + 1}</span>
                        <span className="ranking-cat">{c.category}</span>
                        <div className="ranking-bar">
                          <div
                            className="ranking-bar-fill"
                            style={{ width: `${Math.round(c.score * 100)}%` }}
                          />
                        </div>
                        <span className="ranking-score">{Math.round(c.score * 100)}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              <div className="similar-block">
                <h3>{t.similarTitle}</h3>
                <p className="similar-hint">{t.similarHint}</p>
                <ul className="similar-list">
                  {analysis.similar_papers.map((p) => (
                    <li className="similar-item" key={p.id}>
                      <a href={p.url} target="_blank" rel="noreferrer" className="similar-title">
                        {p.title || p.id}
                      </a>
                      <div className="similar-meta">
                        <span className="similar-cat">{p.category}</span>
                        <span className="similar-sim">
                          {Math.round(p.similarity * 100)}% {t.similarityLabel}
                        </span>
                      </div>
                    </li>
                  ))}
                </ul>
              </div>
            </div>
          ) : null}

          {analysisStatus === 'idle' ? <p className="idle-copy">{t.idleResult}</p> : null}

          <div className="pipeline">
            <h3>{t.pipelineTitle}</h3>
            <div className="pipeline-track">
              {(t.pipeline as string[]).map((item, index) => (
                <div className="pipeline-step" key={item}>
                  <span>{index + 1}</span>
                  <p>{item}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        <aside className="knowledge-panel">
          <InfoCard title={String(t.datasetTitle)} text={String(t.datasetText)} />
          <InfoCard title={String(t.categoriesTitle)} text={String(t.categoriesText)} />
          <InfoCard title={String(t.singleTitle)} text={String(t.singleText)} />
          <InfoCard title={String(t.multiTitle)} text={String(t.multiText)} />
          <InfoCard title={String(t.modelTitle)} text={String(t.modelText)} />
        </aside>
      </section>
    </main>
  );
}

function InfoCard({ title, text }: { title: string; text: string }) {
  return (
    <article className="info-card glass">
      <h3>{title}</h3>
      <p>{text}</p>
    </article>
  );
}

export default App;

