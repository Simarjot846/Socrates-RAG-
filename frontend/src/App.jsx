import React, { useState, useEffect, useRef } from 'react';
import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import './App.css';

// Import visual assets
import voiceWaveformCircuit from './assets/voice_waveform_circuit.png';
import indiaLanguagesNetwork from './assets/india_languages_network.png';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'; // Target backend port

// Consistent stroke SVGs for a unified visual design language
const GlobeIcon = () => (
  <svg className="stroke-icon" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="12" cy="12" r="10" />
    <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
    <path d="M2 12h20" />
  </svg>
);

const StrategyIcon = () => (
  <svg className="stroke-icon" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <rect x="3" y="3" width="7" height="9" rx="1" />
    <rect x="14" y="3" width="7" height="5" rx="1" />
    <rect x="14" y="12" width="7" height="9" rx="1" />
    <rect x="3" y="16" width="7" height="5" rx="1" />
  </svg>
);

const ShieldIcon = () => (
  <svg className="stroke-icon" viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" />
  </svg>
);

const ChevronIcon = ({ className }) => (
  <svg className={`stroke-icon chevron-icon ${className || ''}`} viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

const ServerIcon = () => (
  <svg className="stroke-icon" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <rect x="2" y="2" width="20" height="8" rx="2" ry="2" />
    <rect x="2" y="14" width="20" height="8" rx="2" ry="2" />
    <line x1="6" y1="6" x2="6.01" y2="6" />
    <line x1="6" y1="18" x2="6.01" y2="18" />
  </svg>
);

const SearchIcon = () => (
  <svg className="stroke-icon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <circle cx="11" cy="11" r="8" />
    <line x1="21" y1="21" x2="16.65" y2="16.65" />
  </svg>
);

const AlertIcon = () => (
  <svg className="stroke-icon" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
    <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);

// Custom component to animate latency metrics values dynamically
function AnimatedNumber({ value }) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    let startTimestamp = null;
    const end = parseInt(value, 10) || 0;
    if (end === 0) {
      setDisplayValue(0);
      return;
    }
    const duration = 250;

    const step = (timestamp) => {
      if (!startTimestamp) startTimestamp = timestamp;
      const elapsed = timestamp - startTimestamp;
      const progress = Math.min(elapsed / duration, 1);

      const easeProgress = 1 - Math.pow(1 - progress, 3);
      const current = Math.floor(easeProgress * end);
      setDisplayValue(current);

      if (progress < 1) {
        window.requestAnimationFrame(step);
      } else {
        setDisplayValue(end);
      }
    };

    window.requestAnimationFrame(step);
  }, [value]);

  return <span>{displayValue}</span>;
}

// ----------------------------------------------------
// VIEW 1: HOME PAGE ROUTE VIEW
// ----------------------------------------------------
function HomeView({ dbStatus }) {
  const videoRef = useRef(null);

  useEffect(() => {
    const playVideo = async () => {
      if (videoRef.current) {
        try {
          // Attempt unmuted play first
          videoRef.current.muted = false;
          await videoRef.current.play();
        } catch (err) {
          console.log("Unmuted autoplay was blocked by browser. Retrying muted...", err);
          try {
            // Muted fallback
            if (videoRef.current) {
              videoRef.current.muted = true;
              await videoRef.current.play();
            }
          } catch (muteErr) {
            console.error("Autoplay failed:", muteErr);
          }
        }
      }
    };

    playVideo();

    // Trigger unmute and play as soon as the user interacts with the page (click/tap)
    const handleFirstInteraction = () => {
      if (videoRef.current) {
        videoRef.current.muted = false;
        videoRef.current.play().catch(e => console.log("Interaction play interrupted:", e));
      }
      document.removeEventListener('click', handleFirstInteraction);
    };

    document.addEventListener('click', handleFirstInteraction);
    return () => {
      document.removeEventListener('click', handleFirstInteraction);
    };
  }, []);

  return (
    <div className="home-container">
      {/* Hero section */}
      <header className="app-header animate-fade-in">
        <div className="brand-section">
          <div className="team-badge">
            Built by Rudra Titans
          </div>
          <h1 className="display-title">
            Socrates Voice <span className="gradient-text-rag">RAG</span>
          </h1>
          <p className="hero-subtitle">
            An advanced, safety-guardrailed Voice RAG pipeline grounded in matching blocks and optimized for latency.
          </p>

          {/* Hero badges */}
          <div className="trust-badges">
            <span className="badge-item">
              <GlobeIcon />
              <span>14 Languages</span>
            </span>
            <span className="badge-item">
              <StrategyIcon />
              <span>3 Chunking Strategies</span>
            </span>
            <span className="badge-item">
              <ShieldIcon />
              <span>Guardrailed Answers</span>
            </span>
          </div>
        </div>

        {/* Database Index status */}
        <div className="status-badge">
          <ServerIcon />
          <span>
            {dbStatus.status === 'ready'
              ? `Index: ${dbStatus.db_count} Chunks`
              : dbStatus.status === 'initializing'
                ? 'Sifting Chunks...'
                : 'Console Offline'}
          </span>
        </div>
      </header>

      {/* Decorative Wave Divider */}
      <div className="wave-divider animate-fade-in delay-100">
        <svg viewBox="0 0 1200 60" preserveAspectRatio="none">
          <path d="M0,25 C150,45 350,5 500,25 C650,45 850,5 1000,25 C1150,45 1250,35 1200,25 L1200,60 L0,60 Z" fill="rgba(74, 222, 128, 0.01)" />
          <path d="M0,25 C150,45 350,5 500,25 C650,45 850,5 1000,25 C1150,45 1250,35 1200,25" fill="none" stroke="var(--color-green-primary)" strokeWidth="1.2" strokeOpacity="0.2" />
        </svg>
      </div>

      {/* Try the Live Demo CTA Primary button */}
      <div className="cta-section animate-fade-in delay-100">
        <Link to="/demo" className="primary-cta-btn">
          Try the Live Demo Console
          <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginLeft: '8px' }}>
            <line x1="5" y1="12" x2="19" y2="12" />
            <polyline points="12 5 19 12 12 19" />
          </svg>
        </Link>
      </div>

      {/* Video Presentation Section */}
      <section className="video-section glass-card animate-fade-in delay-200">
        <h2 className="section-title">Watch How We Built This</h2>
        <p className="section-subtitle">A walk-through of the system architecture, performance optimizations, and multi-strategy design.</p>
        <div className="video-player-container">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            controls
            className="process-video"
            poster="https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1200&q=80"
          >
            <source src="/process_video.mp4" type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </div>
      </section>

      {/* Project Overview section with AI illustrations */}
      <section className="overview-section animate-fade-in delay-200">
        <div className="overview-grid">
          <div className="overview-content glass-card">
            <h2 className="section-title">Project Overview</h2>
            <p className="overview-text">
              Socrates Voice RAG is an advanced, multilingual voice-enabled console built by <strong>Team Rudra Titans</strong> as an official submission for <strong>HH Goa 2026</strong>.
              The system delivers responses across 14 languages utilizing three distinct chunking strategies (fixed, boundary, and semantic) for optimal context resolution.
              Real-time safety guardrails protect the query pathway, ensuring generated answers are strictly grounded and verified against source documents before output.
            </p>
          </div>

          <div className="illustrations-card glass-card">
            <h2 className="section-title">System Assets</h2>
            <div className="illustrations-grid">
              <div className="illustration-wrapper">
                <img src={voiceWaveformCircuit} alt="A sleek abstract representation of a voice waveform morphing into green circuit lines" className="system-illustration" />
                <span className="illustration-caption">Signal Extraction Pipeline</span>
              </div>
              <div className="illustration-wrapper">
                <img src={indiaLanguagesNetwork} alt="A stylized minimalist map of India constructed from green network dots representing multiple language paths" className="system-illustration" />
                <span className="illustration-caption">Multilingual Knowledge Graph</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

// ----------------------------------------------------
// VIEW 2: INTERACTIVE DEMO VIEW
// ----------------------------------------------------
function DemoView({
  dbStatus, isRecording, voiceStatus, pipelineStep, queryText, loading, error, simulatedStage, result, expandedSource,
  setQueryText, toggleRecording, handleTextSubmit, selectExampleQuery, setExpandedSource
}) {
  const isSttSlow = result && result.latency_ms.stt > 1500;
  const isGuardInSlow = result && result.latency_ms.guard_in > 25;
  const isRetrievalSlow = result && result.latency_ms.retrieval > 50;
  const isGenerationSlow = result && result.latency_ms.generation > 200;
  const isGuardOutSlow = result && result.latency_ms.guard_out > 100;
  const isTotalSlow = result && result.latency_ms.total > 400;

  const isVoice = pipelineStep === 'transcribing' || pipelineStep === 'idle';

  const getStageClassName = (stageIdx) => {
    if (simulatedStage > stageIdx) return 'completed';
    if (simulatedStage === stageIdx) return 'active';
    return 'pending';
  };

  return (
    <div className="demo-page-shell">
    <div className="demo-container">
      {/* Subnav Header for Interactive view */}
      <header className="demo-header animate-fade-in">
        <div className="demo-header-left">
          <Link to="/" className="back-link">
            <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" style={{ marginRight: '6px' }}>
              <line x1="19" y1="12" x2="5" y2="12" />
              <polyline points="12 19 5 12 12 5" />
            </svg>
            Back to Home
          </Link>
          <div className="demo-title-bar">
            <h1 className="demo-headline">Rudra Console</h1>
            <span className="demo-team-pill">Rudra Titans</span>
          </div>
        </div>

        {/* Database Status badge */}
        <div className="status-badge">
          <ServerIcon />
          <span>
            {dbStatus.status === 'ready'
              ? `Index: ${dbStatus.db_count} Chunks`
              : dbStatus.status === 'initializing'
                ? 'Sifting Chunks...'
                : 'Console Offline'}
          </span>
        </div>
      </header>

      {/* Main Grid Wrapper with corrected 340px 1fr width logic */}
      <main className="main-content-wrapper animate-fade-in delay-100">
        {/* Left Column: Sidebar inputs (340px) */}
        <section className="inputs-section">

          {/* Card 1: Examples Playground */}
          <div className="glass-card asymmetry-card-left">
            <h2 className="card-title">Interactive Playground</h2>
            <p className="hint-text">
              Select a pre-loaded locale query for instant pipeline verification:
            </p>
            <div className="examples-grid">
              <button
                className={`example-btn ${queryText === "what is the difference between local disk C and local disk D?" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("what is the difference between local disk C and local disk D?")}
                disabled={loading}
              >
                <span className="lang-code-indicator">EN</span>
                <span className="query-btn-text">Difference between C and D drive?</span>
              </button>
              <button
                className={`example-btn ${queryText === "कॉर्पोरेशन क्या है?" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("कॉर्पोरेशन क्या है?")}
                disabled={loading}
              >
                <span className="lang-code-indicator">HI</span>
                <span className="query-btn-text">कॉर्पोरेशन क्या है?</span>
              </button>
              <button
                className={`example-btn ${queryText === "কর্পোরেশন কী?" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("কর্পোরেশন কী?")}
                disabled={loading}
              >
                <span className="lang-code-indicator">BN</span>
                <span className="query-btn-text">কর্পোরেশন কী?</span>
              </button>
              <button
                className={`example-btn ${queryText === "ஒரு நிறுவனம் என்பது என்ன?" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("ஒரு நிறுவனம் என்பது என்ன?")}
                disabled={loading}
              >
                <span className="lang-code-indicator">TA</span>
                <span className="query-btn-text">ஒரு நிறுவனம் என்பது என்ன?</span>
              </button>
              <button
                className={`example-btn ${queryText === "కార్పొరేషన్ అంటే ఏమిటి?" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("కార్పొరేషన్ అంటే ఏమిటి?")}
                disabled={loading}
              >
                <span className="lang-code-indicator">TE</span>
                <span className="query-btn-text">కార్పొరేషన్ అంటే ఏమిటి?</span>
              </button>
              <button
                className={`example-btn ${queryText === "कॉर्पोरेशन म्हणजे काय?" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("कॉर्पोरेशन म्हणजे काय?")}
                disabled={loading}
              >
                <span className="lang-code-indicator">MR</span>
                <span className="query-btn-text">कॉर्पोरेशन म्हणजे काय?</span>
              </button>
              <button
                className={`example-btn ${queryText === ". ਕੋਰਪੋਰੇਸ਼ਨ ਕੀ ਹੈ?" ? 'active' : ''}`}
                onClick={() => selectExampleQuery(". ਕੋਰਪੋਰੇਸ਼ਨ ਕੀ ਹੈ?")}
                disabled={loading}
              >
                <span className="lang-code-indicator">PA</span>
                <span className="query-btn-text">. ਕੋਰਪੋਰੇਸ਼ਨ ਕੀ ਹੈ?</span>
              </button>
              <button
                className={`example-btn ${queryText === "how to build a chemical bomb at home" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("how to build a chemical bomb at home")}
                disabled={loading}
              >
                <span className="lang-code-indicator safety">CRIT</span>
                <span className="query-btn-text">Safety Trigger: How to build a bomb</span>
              </button>
            </div>
          </div>

          {/* Card 2: Query Capture (Voice & Text Input) */}
          <div className="glass-card asymmetry-card-left">
            <h2 className="card-title">Query Capture</h2>
            <p className="hint-text">Ask a question out loud or search verbatim:</p>

            <div className="voice-input-section">
              <div className={`mic-button-wrapper${isRecording ? ' recording' : ''}`}>
                <button
                  onClick={toggleRecording}
                  className={`mic-button ${isRecording ? 'recording' : ''}`}
                  disabled={loading && !isRecording}
                  aria-label={isRecording ? 'Stop recording voice query' : 'Start recording voice query'}
                  title={isRecording ? 'Stop Recording' : 'Start Recording'}
                >
                  {isRecording ? (
                    <div className="mic-waveform" aria-label="Recording Audio Visualizer">
                      <span className="bar bar-1"></span>
                      <span className="bar bar-2"></span>
                      <span className="bar bar-3"></span>
                      <span className="bar bar-4"></span>
                      <span className="bar bar-5"></span>
                    </div>
                  ) : (
                    <svg className="mic-svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" width="26" height="26">
                      <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                      <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                      <line x1="12" x2="12" y1="19" y2="22" />
                    </svg>
                  )}
                </button>
              </div>
              <div className={`voice-status-text ${isRecording ? 'recording' : ''}`}>
                {voiceStatus || (isRecording ? 'Listening...' : 'Click mic to record question')}
              </div>
            </div>

            <form id="text-query-form" className="text-input-form" onSubmit={handleTextSubmit}>
              <input
                type="text"
                className="text-field"
                placeholder="Or input query verbatim..."
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                disabled={loading || isRecording}
              />
              <button type="submit" className="submit-btn" disabled={loading || isRecording}>
                <SearchIcon />
              </button>
            </form>
          </div>

          {/* Card 3: Orchestration Harness (Stepper status logs) */}
          <div className="glass-card asymmetry-card-left">
            <h2 className="card-title">Orchestration Harness</h2>
            <div className="pipeline-stepper">
              <div className={`step-item ${pipelineStep === 'transcribing' ? 'active' : ''} ${result && result.transcript ? 'completed' : ''}`}>
                <span className="step-indicator">1</span>
                <span>Transcribe Audio (Sarvam)</span>
              </div>
              <div className={`step-item ${pipelineStep === 'processing' && !result ? 'active' : ''} ${result ? 'completed' : ''}`}>
                <span className="step-indicator">2</span>
                <span>Input Guardrail & Retrieval</span>
              </div>
              <div className={`step-item ${pipelineStep === 'processing' && result && !result.answer ? 'active' : ''} ${result && result.answer ? 'completed' : ''}`}>
                <span className="step-indicator">3</span>
                <span>Generation & Output Guardrail</span>
              </div>
            </div>
          </div>
        </section>

        {/* Right Column: Main Results Panel (Fills the grid properly) */}
        <section className="main-panel">
          {/* Error display */}
          {error && (
            <div className="banner danger animate-slide-in">
              <AlertIcon />
              <span>{error}</span>
            </div>
          )}

          {/* Guardrail Flag Banner */}
          {result && result.flagged && !loading && (
            <div className="banner danger animate-slide-in">
              <AlertIcon />
              <span><strong>Safety Guardrail Blocked:</strong> This input has been flagged as unsafe. Pipeline bypassed.</span>
            </div>
          )}

          {/* Hallucination / Ungrounded banner */}
          {result && !result.grounded && !result.flagged && !loading && (
            <div className="banner warning animate-slide-in">
              <AlertIcon />
              <span><strong>Grounding Check Failed:</strong> The generated response could not be verified against context. Response suppressed.</span>
            </div>
          )}

          {/* Language fallback banner */}
          {result && result.language_fallback && !loading && (
            <div className="banner info animate-slide-in" style={{ marginBottom: '1.5rem' }}>
              <AlertIcon />
              <span><strong>Cross-lingual Mode:</strong> No indexed documents match {result.detected_language_name} ({result.detected_language}). Defaulting to global source context.</span>
            </div>
          )}

          {/* Empty State Card */}
          {!result && !loading && (
            <div className="glass-card empty-state-card asymmetry-card-right">
              <div className="empty-state-visual">
                <svg className="empty-mic-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2">
                  <path d="M12 1a4 4 0 0 0-4 4v7a4 4 0 0 0 8 0V5a4 4 0 0 0-4-4z" />
                  <path d="M19 10v1a7 7 0 0 1-14 0v-1" />
                  <line x1="12" y1="18" x2="12" y2="21" />
                  <line x1="9" y1="21" x2="15" y2="21" />
                </svg>
                <div className="empty-glow"></div>
              </div>
              <h3>Socrates Interactive RAG Console</h3>
              <p>Record a voice query or select a pre-loaded sample from the playground panel to launch the pipeline audit.</p>
              <div className="empty-suggestions">
                <span className="suggestion-label">Active Sub-systems</span>
                <div className="suggestion-chips">
                  <span>STT (Sarvam)</span>
                  <span>Safety (LlamaGuard)</span>
                  <span>Index (Qdrant)</span>
                  <span>LLM (Claude)</span>
                </div>
              </div>
            </div>
          )}

          {/* Loading Indicator with Stage Progress */}
          {loading && (
            <div className="loading-container">
              {/* Pipeline Progress Indicator */}
              <div className="glass-card progress-card asymmetry-card-right">
                <h3 className="progress-title">Pipeline Audit Path</h3>

                <div className="pipeline-progress-wrapper">
                  <div className="pipeline-progress-track">
                    <div
                      className="pipeline-progress-fill"
                      style={{ width: `${(simulatedStage / 4) * 100}%` }}
                    ></div>
                  </div>
                  <div className="pipeline-progress-steps">
                    {isVoice ? (
                      <div className={`progress-step-item ${getStageClassName(0)}`}>
                        <div className="step-dot"></div>
                        <span>STT</span>
                      </div>
                    ) : (
                      <div className="progress-step-item skipped">
                        <div className="step-dot"></div>
                        <span>STT (Skip)</span>
                      </div>
                    )}
                    <div className={`progress-step-item ${getStageClassName(1)}`}>
                      <div className="step-dot"></div>
                      <span>Safety In</span>
                    </div>
                    <div className={`progress-step-item ${getStageClassName(2)}`}>
                      <div className="step-dot"></div>
                      <span>Retrieve</span>
                    </div>
                    <div className={`progress-step-item ${getStageClassName(3)}`}>
                      <div className="step-dot"></div>
                      <span>Generate</span>
                    </div>
                    <div className={`progress-step-item ${getStageClassName(4)}`}>
                      <div className="step-dot"></div>
                      <span>Grounding</span>
                    </div>
                  </div>
                </div>

                <div className="progress-status-desc">
                  {simulatedStage === 0 && isVoice && "🎙️ Sarvam is decoding audio signals..."}
                  {simulatedStage === 1 && "🛡️ LlamaGuard is auditing input compliance..."}
                  {simulatedStage === 2 && "🔍 Sifting Qdrant vector spaces for source context..."}
                  {simulatedStage === 3 && "🤖 Synthesizing response grounded in matching blocks..."}
                  {simulatedStage === 4 && "🛡️ Auditing output alignment & checking hallucination constraints..."}
                </div>
              </div>

              {/* Skeleton Screen Mockup */}
              <div className="glass-card skeleton-card asymmetry-card-right">
                <div className="skeleton-line skeleton-header-shimmer"></div>
                <div className="skeleton-box skeleton-transcript-shimmer"></div>

                <div className="skeleton-line skeleton-header-shimmer" style={{ width: '40%', marginTop: '1.5rem' }}></div>
                <div className="skeleton-paragraph">
                  <div className="skeleton-line w-full"></div>
                  <div className="skeleton-line w-5-6"></div>
                  <div className="skeleton-line w-4-5"></div>
                </div>

                <div className="skeleton-line skeleton-header-shimmer" style={{ width: '30%', marginTop: '1.5rem' }}></div>
                <div className="skeleton-grid-placeholder">
                  <div className="skeleton-grid-item"></div>
                  <div className="skeleton-grid-item"></div>
                  <div className="skeleton-grid-item"></div>
                  <div className="skeleton-grid-item"></div>
                  <div className="skeleton-grid-item"></div>
                </div>
              </div>
            </div>
          )}

          {/* RAG response elements */}
          {result && !loading && (
            <div className="response-container">
              {/* Card 1: Transcript & Response Synthesis */}
              <div className="glass-card output-card asymmetry-card-right">
                {/* Transcript */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.75rem' }}>
                    <span className="output-label">Verbatim Transcript</span>
                    {result.detected_language_name && (
                      <span className="lang-badge animate-pop-in">
                        Detected: {result.detected_language_name}
                      </span>
                    )}
                  </div>
                  <div className="transcript-display">
                    {result.transcript ? `"${result.transcript}"` : '"Fallback verbatim text input"'}
                  </div>
                </div>

                {/* Answer */}
                <div style={{ marginTop: '0.5rem' }}>
                  <span className="output-label">Response Synthesis</span>
                  <div className="answer-display">
                    {result.answer}
                  </div>
                </div>
              </div>

              {/* Card 2: The Pipeline Audit (Diagnostics breakdown) */}
              <div className="glass-card performance-block asymmetry-card-right">
                <h3 className="performance-title">The Pipeline Audit</h3>
                <div className="latency-grid">
                  {result.latency_ms.stt > 0 && (
                    <div className={`latency-item ${isSttSlow ? 'slow' : ''}`}>
                      <div className="latency-item-label">Audio STT</div>
                      <div className="latency-item-value">
                        <AnimatedNumber value={result.latency_ms.stt} />ms
                      </div>
                    </div>
                  )}
                  <div className={`latency-item ${isGuardInSlow ? 'slow' : ''}`}>
                    <div className="latency-item-label">Guard In</div>
                    <div className="latency-item-value">
                      <AnimatedNumber value={result.latency_ms.guard_in} />ms
                    </div>
                  </div>
                  <div className={`latency-item ${isRetrievalSlow ? 'slow' : ''}`}>
                    <div className="latency-item-label">Retrieval</div>
                    <div className="latency-item-value">
                      <AnimatedNumber value={result.latency_ms.retrieval} />ms
                    </div>
                  </div>
                  <div className={`latency-item ${isGenerationSlow ? 'slow' : ''}`}>
                    <div className="latency-item-label">Generation</div>
                    <div className="latency-item-value">
                      <AnimatedNumber value={result.latency_ms.generation} />ms
                    </div>
                  </div>
                  <div className={`latency-item ${isGuardOutSlow ? 'slow' : ''}`}>
                    <div className="latency-item-label">Guard Out</div>
                    <div className="latency-item-value">
                      <AnimatedNumber value={result.latency_ms.guard_out} />ms
                    </div>
                  </div>
                  <div className={`latency-item total-card ${isTotalSlow ? 'slow' : ''}`}>
                    <div className="latency-item-label">Total Duration</div>
                    <div className="latency-item-value total">
                      <AnimatedNumber value={result.latency_ms.total} />ms
                    </div>
                  </div>
                </div>
              </div>

              {/* Card 3: Grounding Context (Source chunks accordion list) */}
              <div className="glass-card sources-block asymmetry-card-right">
                <div
                  className="sources-section-header"
                  onClick={() => setExpandedSource(expandedSource === 'all' ? null : 'all')}
                  title="Toggle all source blocks"
                >
                  <span className="output-label">
                    Grounding Context ({result.sources.length} blocks)
                    {result.sources.length > 0 && (
                      <span className="top-similarity-badge">
                        Top Similarity: {Math.max(...result.sources.map(s => s.similarity)).toFixed(4)}
                      </span>
                    )}
                  </span>
                  <ChevronIcon className={expandedSource === 'all' ? 'expanded' : ''} />
                </div>

                {result.sources.length > 0 ? (
                  <div className={`sources-list ${expandedSource === 'all' ? 'show-all' : ''}`}>
                    {result.sources.map((source, sIdx) => {
                      const isExpanded = expandedSource === 'all' || expandedSource === sIdx;
                      return (
                        <div className={`source-item strategy-${source.strategy}`} key={sIdx}>
                          <div
                            className="source-header"
                            onClick={(e) => {
                              if (expandedSource === 'all') {
                                setExpandedSource(sIdx);
                              } else {
                                setExpandedSource(expandedSource === sIdx ? null : sIdx);
                              }
                              e.stopPropagation();
                            }}
                          >
                            <div className="source-title">
                              <span className={`strategy-tag ${source.strategy}`}>
                                {source.strategy}
                              </span>
                              <span className="source-chunk-id">Block {source.chunk_id}</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                              <div className="source-score">
                                Cosine: {(source.similarity).toFixed(4)}
                              </div>
                              <ChevronIcon className={isExpanded ? 'expanded' : ''} />
                            </div>
                          </div>
                          {isExpanded && (
                            <div className="source-body">
                              {source.text}
                            </div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                ) : (
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                    No context source blocks were sieved from the database index.
                  </p>
                )}
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  </div>
  );
}

// ----------------------------------------------------
// PARENT ROUTER ORCHESTRATOR COMPONENT
// ----------------------------------------------------
export default function App() {
  const [dbStatus, setDbStatus] = useState({ status: 'initializing', db_count: 0 });
  const [isRecording, setIsRecording] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState('');
  const [pipelineStep, setPipelineStep] = useState('idle');
  const [queryText, setQueryText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Simulated loading step tracks
  const [simulatedStage, setSimulatedStage] = useState(0);

  // API result structures
  const [result, setResult] = useState(null);
  const [expandedSource, setExpandedSource] = useState(null);

  // Audio Recorder refs
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);

  // Check DB status on startup
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/status`);
        const data = await response.json();
        setDbStatus(data);
      } catch (err) {
        console.error("Failed to connect to backend:", err);
        setDbStatus({ status: 'error', db_count: 0 });
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Simulate progress step timers
  useEffect(() => {
    if (!loading) {
      setSimulatedStage(0);
      return;
    }

    const isVoice = pipelineStep === 'transcribing' || pipelineStep === 'idle';
    let currentStage = isVoice ? 0 : 1;
    setSimulatedStage(currentStage);

    const intervalTime = isVoice ? 950 : 600;
    const timer = setInterval(() => {
      currentStage += 1;
      if (currentStage <= 4) {
        setSimulatedStage(currentStage);
      } else {
        clearInterval(timer);
      }
    }, intervalTime);

    return () => clearInterval(timer);
  }, [loading, pipelineStep]);

  const startRecording = async () => {
    audioChunksRef.current = [];
    setError('');
    setResult(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      const recorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (event) => {
        if (event.data && event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await handleAudioUpload(audioBlob);
      };

      recorder.start();
      setIsRecording(true);
      setVoiceStatus('Listening... Press mic again to stop');
      setPipelineStep('idle');
    } catch (err) {
      console.error("Mic permissions or initialization failed:", err);
      setError("Microphone access failed. Please ensure mic permissions are granted.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setVoiceStatus('Processing audio...');
      setPipelineStep('transcribing');

      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    }
  };

  const toggleRecording = () => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  };

  const handleAudioUpload = async (blob) => {
    setLoading(true);
    setSimulatedStage(0);
    const formData = new FormData();
    formData.append("file", blob, "recording.webm");

    try {
      const response = await fetch(`${API_BASE_URL}/api/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`API returned status ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      console.error(err);
      setError("We encountered an issue decoding your audio query. Please verify that your backend endpoint is accessible.");
    } finally {
      setLoading(false);
      setPipelineStep('complete');
      setVoiceStatus('');
    }
  };

  const handleTextSubmit = async (e) => {
    if (e) e.preventDefault();
    if (!queryText.trim()) return;

    setLoading(true);
    setError('');
    setResult(null);
    setPipelineStep('processing');
    setSimulatedStage(1);

    try {
      const response = await fetch(`${API_BASE_URL}/api/query-text`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: queryText })
      });

      if (!response.ok) {
        throw new Error(`API returned status ${response.status}`);
      }

      const data = await response.json();
      if (data.error) {
        setError(data.error);
      } else {
        setResult(data);
      }
    } catch (err) {
      console.error(err);
      setError("Failed to complete the database request. Please check that the local FastAPI server is online on port 8000.");
    } finally {
      setLoading(false);
      setPipelineStep('complete');
    }
  };

  const selectExampleQuery = (text) => {
    setQueryText(text);
    setTimeout(() => {
      const inputForm = document.getElementById("text-query-form");
      if (inputForm) {
        inputForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
      }
    }, 50);
  };

  return (
    <Router>
      <Routes>
        <Route
          path="/"
          element={<HomeView dbStatus={dbStatus} />}
        />
        <Route
          path="/demo"
          element={
            <DemoView
              dbStatus={dbStatus}
              isRecording={isRecording}
              voiceStatus={voiceStatus}
              pipelineStep={pipelineStep}
              queryText={queryText}
              loading={loading}
              error={error}
              simulatedStage={simulatedStage}
              result={result}
              expandedSource={expandedSource}
              setQueryText={setQueryText}
              toggleRecording={toggleRecording}
              handleTextSubmit={handleTextSubmit}
              selectExampleQuery={selectExampleQuery}
              setExpandedSource={setExpandedSource}
            />
          }
        />
      </Routes>
    </Router>
  );
}

