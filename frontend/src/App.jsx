import React, { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE_URL = 'http://localhost:8000'; // Target backend port

export default function App() {
  const [dbStatus, setDbStatus] = useState({ status: 'initializing', db_count: 0 });
  const [isRecording, setIsRecording] = useState(false);
  const [voiceStatus, setVoiceStatus] = useState('');
  const [pipelineStep, setPipelineStep] = useState('idle'); // idle, transcribing, processing, complete
  const [queryText, setQueryText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  
  // Results
  const [result, setResult] = useState(null);
  const [expandedSource, setExpandedSource] = useState(null);

  // Refs for audio recorder
  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const streamRef = useRef(null);

  // 1. Fetch DB Status on load
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

  // 2. Microphone Recording Logic
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
      
      // Stop all tracks in stream to release mic indicator
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

  // 3. Backend Communications
  const handleAudioUpload = async (blob) => {
    setLoading(true);
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
      setError("Failed to process voice query. Please try again.");
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
    setPipelineStep('processing'); // text query bypasses transcribing step

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
      setError("Failed to execute query. Please check backend connection.");
    } finally {
      setLoading(false);
      setPipelineStep('complete');
    }
  };

  const selectExampleQuery = (text) => {
    setQueryText(text);
    // Submit text query in next tick
    setTimeout(() => {
      const inputForm = document.getElementById("text-query-form");
      if (inputForm) {
        inputForm.dispatchEvent(new Event('submit', { cancelable: true, bubbles: true }));
      }
    }, 50);
  };

  // Determine if specific latencies exceed their budget targets
  const isSttSlow = result && result.latency_ms.stt > 1500;
  const isGuardInSlow = result && result.latency_ms.guard_in > 25;
  const isRetrievalSlow = result && result.latency_ms.retrieval > 50;
  const isGenerationSlow = result && result.latency_ms.generation > 200;
  const isGuardOutSlow = result && result.latency_ms.guard_out > 100;
  const isTotalSlow = result && result.latency_ms.total > 400;

  return (
    <div className="app-container">
      {/* Header */}
      <header className="app-header">
        <div className="brand-section">
          <h1 className="display-title">
            Socrates Voice <span className="gradient-text-rag">RAG</span>
            <svg className="wave-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M2 12c4-4 6-4 10 0s6 4 10 0" strokeLinecap="round"/>
              <path d="M2 16c4-4 6-4 10 0s6 4 10 0" strokeLinecap="round"/>
            </svg>
          </h1>
          <p>Multi-Strategy Chunking & Guardrailed RAG Web Console</p>
        </div>
        <div className="status-badge">
          <span className={`status-dot ${dbStatus.status}`}></span>
          <span>
            {dbStatus.status === 'ready' 
              ? `DB Count: ${dbStatus.db_count} chunks` 
              : dbStatus.status === 'initializing' 
                ? 'Initializing DB...' 
                : 'Server offline'}
          </span>
        </div>
      </header>

      {/* Main Panel Grid */}
      <main className="main-grid">
        {/* Left Side: Inputs */}
        <section className="inputs-section">
          <div className="glass-card" style={{ marginBottom: '1.5rem' }}>
            <h2 className="card-title">Demo Dashboard</h2>
            
            {/* Quick Demo Examples */}
            <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.75rem' }}>
              Select a pre-loaded query for instant testing:
            </p>
            <div className="examples-grid">
              <button 
                className={`example-btn ${queryText === "what is the difference between local disk C and local disk D?" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("what is the difference between local disk C and local disk D?")}
                disabled={loading}
              >
                🇬🇧 English: Difference between C and D drive?
              </button>
              <button 
                className={`example-btn ${queryText === "पोटेशियम में कम खाद्य पदार्थों का चार्ट।" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("पोटेशियम में कम खाद्य पदार्थों का चार्ट।")}
                disabled={loading}
              >
                🇮🇳 Hindi: पोटेशियम में कम खाद्य पदार्थों का चार्ट।
              </button>
              <button 
                className={`example-btn ${queryText === "একটি কর্পোরেশন কি?" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("একটি কর্পোরেশন কি?")}
                disabled={loading}
              >
                🇧🇩 Bengali: একটি কর্পোরেশন কি? (What is a corporation?)
              </button>
              <button 
                className={`example-btn ${queryText === "சூரிய மின் தகடுகள் எவ்வளவு கி.வா மின்சாரம் உற்பத்தி செய்கின்றன" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("சூரிய மின் தகடுகள் எவ்வளவு கி.வா மின்சாரம் உற்பத்தி செய்கின்றன")}
                disabled={loading}
              >
                🇮🇳 Tamil: சூரிய மின் தகடுகள் எவ்வளவு கி.வா மின்சாரம்...
              </button>
              <button 
                className={`example-btn ${queryText === "కార్పొరేషన్ అంటే ఏమిటి?" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("కార్పొరేషన్ అంటే ఏమిటి?")}
                disabled={loading}
              >
                🇮🇳 Telugu: కార్పొరేషన్ అంటే ఏమిటి? (What is a corporation?)
              </button>
              <button 
                className={`example-btn ${queryText === "पोटॅशियम कमी असलेल्या पदार्थांचा तक्ता." ? 'active' : ''}`}
                onClick={() => selectExampleQuery("पोटॅशियम कमी असलेल्या पदार्थांचा तक्ता.")}
                disabled={loading}
              >
                🇮🇳 Marathi: पोटॅशियम कमी असलेल्या पदार्थांचा तक्ता.
              </button>
              <button 
                className={`example-btn ${queryText === "ਸੂਰਜੀ ਪੈਨਲ ਕਿੰਨੇ ਕਿਲੋਵਾਟ ਬਿਜਲੀ ਪੈਦਾ ਕਰਦੇ ਹਨ?" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("ਸੂਰਜੀ ਪੈਨਲ ਕਿੰਨੇ ਕਿਲੋਵਾਟ ਬਿਜਲੀ ਪੈਦਾ ਕਰਦੇ ਹਨ?")}
                disabled={loading}
              >
                🇮🇳 Punjabi: ਸੂਰਜੀ ਪੈਨਲ ਕਿੰਨੇ ਕਿਲੋਵਾਟ ਬਿਜਲੀ ਪੈਦਾ...
              </button>
              <button 
                className={`example-btn ${queryText === "how to build a chemical bomb at home" ? 'active' : ''}`}
                onClick={() => selectExampleQuery("how to build a chemical bomb at home")}
                disabled={loading}
                style={{ borderLeft: '3px solid var(--color-coral)' }}
              >
                ⚠️ Safety: How to build a bomb
              </button>
            </div>

            {/* Voice Input */}
            <div className="voice-input-section">
              <div className="mic-button-wrapper">
                <button 
                  onClick={toggleRecording} 
                  className={`mic-button ${isRecording ? 'recording' : ''}`}
                  disabled={loading && !isRecording}
                  title={isRecording ? 'Stop Recording' : 'Start Recording'}
                >
                  {isRecording ? '⏹️' : '🎤'}
                </button>
              </div>
              <div className={`voice-status-text ${isRecording ? 'recording' : ''}`}>
                {voiceStatus || (isRecording ? 'Listening...' : 'Click mic to record question')}
              </div>
            </div>

            {/* Fallback Text Input */}
            <form id="text-query-form" className="text-input-form" onSubmit={handleTextSubmit}>
              <input 
                type="text" 
                className="text-field"
                placeholder="Or type your question here..."
                value={queryText}
                onChange={(e) => setQueryText(e.target.value)}
                disabled={loading || isRecording}
              />
              <button type="submit" className="submit-btn" disabled={loading || isRecording}>
                Send
              </button>
            </form>
          </div>

          {/* Stepper tracking stage */}
          <div className="glass-card">
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

        {/* Right Side: Results Display */}
        <section className="results-section">
          {/* Error display */}
          {error && (
            <div className="banner danger">
              <span className="banner-icon font-pulse">⚠️</span>
              <span>{error}</span>
            </div>
          )}

          {/* Guardrail Flag Banner */}
          {result && result.flagged && (
            <div className="banner danger">
              <span className="banner-icon font-pulse">🚨</span>
              <span><strong>Query Blocked:</strong> Input flagged by safety guardrails. Bypassed indexing and generation.</span>
            </div>
          )}

          {/* Hallucination / Ungrounded banner */}
          {result && !result.grounded && !result.flagged && (
            <div className="banner warning">
              <span className="banner-icon font-pulse">⚠️</span>
              <span><strong>Hallucination Guardrail Triggered:</strong> Generated answer was ungrounded by context, response blocked.</span>
            </div>
          )}

          {/* Language fallback banner */}
          {result && result.language_fallback && (
            <div className="banner info" style={{ marginBottom: '1rem' }}>
              <span className="banner-icon">ℹ️</span>
              <span><strong>Limited Data:</strong> No direct context found for {result.detected_language_name} ({result.detected_language}). Showing best-effort results.</span>
            </div>
          )}

          {/* RAG response */}
          {result ? (
            <div className="glass-card output-card">
              {/* Transcript */}
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span className="output-label">Question Transcript</span>
                  {result.detected_language_name && (
                    <span className="lang-badge">
                      Detected: {result.detected_language_name} ({result.detected_language})
                    </span>
                  )}
                </div>
                <div className="transcript-display">
                  {result.transcript || '"Text query fallback"'}
                </div>
              </div>

              {/* Answer */}
              <div>
                <span className="output-label">Answer Response</span>
                <div className="answer-display" style={{ color: result.grounded ? 'var(--text-primary)' : 'var(--color-orange)' }}>
                  {result.answer}
                </div>
              </div>

              {/* Latency Dashboard */}
              <div style={{ marginTop: '1rem', borderTop: '1px solid var(--border-color)', paddingTop: '1.25rem' }}>
                <span className="output-label">Latency Breakdown (Exclude Upload)</span>
                <div className="latency-grid">
                  {result.latency_ms.stt > 0 && (
                    <div className={`latency-item ${isSttSlow ? 'slow' : ''}`}>
                      <div className="latency-item-label">STT</div>
                      <div className="latency-item-value">{result.latency_ms.stt}ms</div>
                    </div>
                  )}
                  <div className={`latency-item ${isGuardInSlow ? 'slow' : ''}`}>
                    <div className="latency-item-label">Guard In</div>
                    <div className="latency-item-value">{result.latency_ms.guard_in}ms</div>
                  </div>
                  <div className={`latency-item ${isRetrievalSlow ? 'slow' : ''}`}>
                    <div className="latency-item-label">Retrieval</div>
                    <div className="latency-item-value">{result.latency_ms.retrieval}ms</div>
                  </div>
                  <div className={`latency-item ${isGenerationSlow ? 'slow' : ''}`}>
                    <div className="latency-item-label">Generation</div>
                    <div className="latency-item-value">{result.latency_ms.generation}ms</div>
                  </div>
                  <div className={`latency-item ${isGuardOutSlow ? 'slow' : ''}`}>
                    <div className="latency-item-label">Guard Out</div>
                    <div className="latency-item-value">{result.latency_ms.guard_out}ms</div>
                  </div>
                  <div className={`latency-item total-card ${isTotalSlow ? 'slow' : ''}`} style={{ borderLeft: 'none' }}>
                    <div className="latency-item-label">Total</div>
                    <div className="latency-item-value total">{result.latency_ms.total}ms</div>
                  </div>
                </div>
              </div>

              {/* Retrieved Chunks Accordion */}
              <div style={{ marginTop: '1rem', borderTop: '1px solid var(--border-color)', paddingTop: '1.25rem' }}>
                <span className="output-label">Retrieved Source Chunks ({result.sources.length})</span>
                {result.sources.length > 0 ? (
                  <div className="sources-list">
                    {result.sources.map((source, sIdx) => (
                      <div className={`source-item strategy-${source.strategy}`} key={sIdx}>
                        <div 
                          className="source-header" 
                          onClick={() => setExpandedSource(expandedSource === sIdx ? null : sIdx)}
                        >
                          <div className="source-title">
                            <span className={`strategy-tag ${source.strategy}`}>
                              {source.strategy}
                            </span>
                            <span>{source.chunk_id}</span>
                          </div>
                          <div className="source-score">
                            Cosine Similarity: {(source.similarity).toFixed(4)}
                          </div>
                        </div>
                        {expandedSource === sIdx && (
                          <div className="source-body">
                            {source.text}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p style={{ fontSize: '0.9rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                    No context chunks retrieved.
                  </p>
                )}
              </div>
            </div>
          ) : (
            !loading && (
              <div className="glass-card empty-state">
                <div className="empty-state-icon">🎙️</div>
                <h3>Awaiting Input</h3>
                <p>Record a voice question or select a demo example from the left panel.</p>
              </div>
            )
          )}

          {/* Loading Indicator */}
          {loading && (
            <div className="glass-card empty-state loading-state">
              <div className="empty-state-icon wave-loader">
                <span className="dot dot-1"></span>
                <span className="dot dot-2"></span>
                <span className="dot dot-3"></span>
              </div>
              <h3>Running Orchestration Harness</h3>
              <p>Executing RAG Pipeline stages...</p>
              <div className="loading-bar-wrapper">
                <div className="loading-bar"></div>
              </div>
            </div>
          )}
        </section>
      </main>
    </div>
  );
}
