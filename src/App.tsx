import { ChangeEvent, FormEvent, useMemo, useState } from 'react';

type Language = 'pt' | 'en';
type ResultState = 'idle' | 'missing-input' | 'pending-integration';
type ArxivStatus = 'idle' | 'loading' | 'success' | 'error';

const copy = {
  pt: {
    eyebrow: 'Demonstração frontend',
    title: 'Arxiv Classifier',
    subtitle:
      'Uma interface acadêmica para classificar artigos científicos a partir do abstract, preparada para receber o motor de inferência nas próximas etapas.',
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
    uploadHint: 'Selecione um arquivo PDF para a futura etapa de extração de texto.',
    uploadButton: 'Escolher PDF',
    noFile: 'Nenhum arquivo selecionado',
    fileReady: 'Arquivo pronto para demonstração',
    abstractTitle: 'Colar abstract',
    abstractHint:
      'Cole o resumo do artigo para preparar a entrada que será enviada ao pipeline de classificação.',
    abstractPlaceholder:
      'Ex.: We propose a neural approach for representation learning in scientific documents...',
    chars: 'caracteres',
    analyze: 'Preparar análise',
    validation: 'Envie um PDF, cole um abstract ou insira um link do arXiv antes de preparar a análise.',
    resultTitle: 'Retorno do sistema',
    idleResult:
      'A área de retorno está pronta. Quando a inferência for integrada, ela exibirá categorias amplas do arXiv e indicadores de confiança.',
    pendingTitle: 'Motor de inferência em desenvolvimento',
    pendingText:
      'Entrada recebida no frontend. A extração de PDF, o pré-processamento e a predição real serão conectados na próxima fase do projeto.',
    honestLabel: 'Sem predição simulada',
    pipelineTitle: 'Pipeline planejado',
    pipeline: ['Entrada do paper', 'Pré-processamento', 'Representação vetorial', 'Classificação'],
    datasetTitle: 'Base de conhecimento',
    datasetText:
      'O projeto usa o arXiv Dataset da Cornell University como fonte de abstracts, títulos, autores e categorias.',
    categoriesTitle: 'Categorias amplas',
    categoriesText:
      'A primeira versão classifica no nível principal, como cs, math e physics, deixando subáreas para uma evolução futura.',
    singleTitle: 'Single-label',
    singleText:
      'Avalia uma categoria principal por artigo com acurácia, precisão, recall, F1 e matriz de confusão.',
    multiTitle: 'Multi-label',
    multiText:
      'Permite múltiplas categorias por artigo e compara F1 micro/macro com Hamming loss.',
    modelTitle: 'Modelos futuros',
    modelText:
      'A comparação planejada inclui baseline TF-IDF com classificador linear e uma rede neural selecionada pelas métricas.',
    sampleTags: ['cs', 'math', 'physics', 'q-bio', 'stat'],
  },
  en: {
    eyebrow: 'Frontend demo',
    title: 'Arxiv Classifier',
    subtitle:
      'An academic interface for classifying scientific papers from abstracts, ready to receive the inference engine in the next stages.',
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
    uploadHint: 'Select a PDF file for the future text extraction step.',
    uploadButton: 'Choose PDF',
    noFile: 'No file selected',
    fileReady: 'File ready for demo',
    abstractTitle: 'Paste abstract',
    abstractHint:
      'Paste the paper abstract to prepare the input that will later be sent to the classification pipeline.',
    abstractPlaceholder:
      'Example: We propose a neural approach for representation learning in scientific documents...',
    chars: 'characters',
    analyze: 'Prepare analysis',
    validation: 'Upload a PDF, paste an abstract, or enter an arXiv link before preparing the analysis.',
    resultTitle: 'System output',
    idleResult:
      'The output area is ready. Once inference is integrated, it will display broad arXiv categories and confidence indicators.',
    pendingTitle: 'Inference engine in development',
    pendingText:
      'Input received by the frontend. PDF extraction, preprocessing, and real prediction will be connected in the next project phase.',
    honestLabel: 'No simulated prediction',
    pipelineTitle: 'Planned pipeline',
    pipeline: ['Paper input', 'Preprocessing', 'Vector representation', 'Classification'],
    datasetTitle: 'Knowledge base',
    datasetText:
      'The project uses the Cornell University arXiv Dataset as a source of abstracts, titles, authors, and categories.',
    categoriesTitle: 'Broad categories',
    categoriesText:
      'The first version classifies at the main level, such as cs, math, and physics, keeping subareas for a future evolution.',
    singleTitle: 'Single-label',
    singleText:
      'Evaluates one main category per paper with accuracy, precision, recall, F1, and confusion matrix.',
    multiTitle: 'Multi-label',
    multiText:
      'Allows multiple categories per paper and compares micro/macro F1 with Hamming loss.',
    modelTitle: 'Future models',
    modelText:
      'The planned comparison includes a TF-IDF baseline with a linear classifier and a neural network selected by metrics.',
    sampleTags: ['cs', 'math', 'physics', 'q-bio', 'stat'],
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
  const [abstractText, setAbstractText] = useState('');
  const [resultState, setResultState] = useState<ResultState>('idle');
  const [arxivUrl, setArxivUrl] = useState('');
  const [arxivStatus, setArxivStatus] = useState<ArxivStatus>('idle');
  const [arxivMessage, setArxivMessage] = useState('');

  const t = copy[language];
  const hasInput = Boolean(fileName || abstractText.trim());
  const abstractCount = useMemo(() => abstractText.trim().length, [abstractText]);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    setFileName(file?.name ?? '');
    setResultState('idle');
  }

  function handleAbstractChange(event: ChangeEvent<HTMLTextAreaElement>) {
    setAbstractText(event.target.value);
    setResultState('idle');
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setResultState(hasInput ? 'pending-integration' : 'missing-input');
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
      const res = await fetch(`https://export.arxiv.org/api/query?id_list=${id}`);
      const text = await res.text();
      const doc = new DOMParser().parseFromString(text, 'application/xml');
      const summary = doc.querySelector('entry summary');
      if (!summary?.textContent?.trim()) throw new Error('not found');
      setAbstractText(summary.textContent.trim());
      setArxivStatus('success');
      setArxivMessage(String(t.arxivSuccess));
      setResultState('idle');
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
            <div className="paper-equation">{'TF-IDF -> Linear model | Neural network'}</div>
            <div className="paper-tags">
              <span>cs.LG</span>
              <span>math.ST</span>
              <span>stat.ML</span>
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

          {fileName ? <div className="status-pill success">{t.fileReady}</div> : null}

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
            <button type="submit">{t.analyze}</button>
          </div>
        </form>

        <section className="result-panel glass" aria-live="polite">
          <div className="result-header">
            <span className="section-index">04</span>
            <h2>{t.resultTitle}</h2>
          </div>

          {resultState === 'missing-input' ? (
            <div className="message warning">{t.validation}</div>
          ) : null}

          {resultState === 'pending-integration' ? (
            <div className="development-state">
              <span className="status-pill neutral">{t.honestLabel}</span>
              <h3>{t.pendingTitle}</h3>
              <p>{t.pendingText}</p>
            </div>
          ) : (
            <p className="idle-copy">{t.idleResult}</p>
          )}

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
