import { useState, useRef } from 'react'
import { 
  Upload, 
  ChevronRight, 
  BookOpen, 
  AlertCircle, 
  Award, 
  CheckCircle, 
  XCircle, 
  RefreshCw, 
  Layers, 
  TrendingUp, 
  TrendingDown, 
  ClipboardList,
  Sparkles,
  User,
  Activity,
  Briefcase,
  Building2,
  FileText
} from 'lucide-react'
import './App.css'

const BACKEND_URL = "http://localhost:8000"

function App() {
  // Navigation & session states
  const [view, setView] = useState('landing') // landing | ready | interview | feedback | summary
  const [role, setRole] = useState('ai_ml_engineer') // ai_ml_engineer | data_science_applied_ml
  const [file, setFile] = useState(null)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  
  // Data states
  const [session, setSession] = useState(null)
  const [currentQuestion, setCurrentQuestion] = useState(null)
  const [answerText, setAnswerText] = useState('')
  const [submittingAnswer, setSubmittingAnswer] = useState(false)
  const [feedback, setFeedback] = useState(null)
  const [summary, setSummary] = useState(null)
  const [loadingQuestion, setLoadingQuestion] = useState(false)

  // Optional Target Company and Job Description (JD) states
  const [company, setCompany] = useState('')
  const [jdText, setJdText] = useState('')
  const [jdFile, setJdFile] = useState(null)
  const [showJdInput, setShowJdInput] = useState(false)
  const jdFileInputRef = useRef(null)

  // Drag and drop ref
  const [dragActive, setDragActive] = useState(false)
  const fileInputRef = useRef(null)

  // Drag handlers
  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true)
    } else if (e.type === "dragleave") {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const droppedFile = e.dataTransfer.files[0]
      if (droppedFile.name.endsWith('.pdf') || droppedFile.name.endsWith('.txt')) {
        setFile(droppedFile)
        setError(null)
      } else {
        setError('Only .pdf and .txt files are supported.')
      }
    }
  }

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      if (selectedFile.name.endsWith('.pdf') || selectedFile.name.endsWith('.txt')) {
        setFile(selectedFile)
        setError(null)
      } else {
        setError('Only .pdf and .txt files are supported.')
      }
    }
  }

  const onButtonClick = () => {
    fileInputRef.current.click()
  }

  const handleJdFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0]
      if (selectedFile.name.endsWith('.txt')) {
        const reader = new FileReader()
        reader.onload = (event) => {
          setJdText(event.target.result)
          setJdFile(selectedFile)
        }
        reader.readAsText(selectedFile)
      } else if (selectedFile.name.endsWith('.pdf')) {
        setJdFile(selectedFile)
        setJdText(`[Uploaded JD PDF: ${selectedFile.name}]`)
      } else {
        setError('Only .pdf and .txt files are supported for Job Descriptions.')
      }
    }
  }

  // API Call: Start Interview
  const handleUploadAndStart = async () => {
    if (!file) {
      setError('Please select or upload a resume file.')
      return
    }

    setUploading(true)
    setError(null)

    const formData = new FormData()
    formData.append('role', role)
    formData.append('file', file)
    if (company && company.trim()) {
      formData.append('company', company.trim())
    }
    if (jdText && jdText.trim()) {
      formData.append('jd_text', jdText.trim())
    }

    try {
      const response = await fetch(`${BACKEND_URL}/session/start`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(errorText || 'Failed to start interview session.')
      }

      const data = await response.json()
      setSession(data)
      setView('ready')
    } catch (err) {
      console.error(err)
      setError(err.message || 'Server error. Please ensure the backend is running.')
    } finally {
      setUploading(false)
    }
  }

  // API Call: Fetch Next Question
  const fetchNextQuestion = async () => {
    if (!session) return
    setError(null)
    setLoadingQuestion(true)
    
    try {
      const response = await fetch(`${BACKEND_URL}/session/${session.session_id}/next-question`)
      if (!response.ok) {
        throw new Error('Failed to retrieve the next question.')
      }

      const data = await response.json()
      if (data.interview_complete) {
        // Fetch session summary and transition to summary screen
        fetchSummary()
      } else {
        setCurrentQuestion(data)
        setAnswerText('')
        setFeedback(null)
        setView('interview')
      }
    } catch (err) {
      console.error(err)
      setError(err.message)
    } finally {
      setLoadingQuestion(false)
    }
  }

  // API Call: Submit Answer
  const submitAnswer = async () => {
    if (!answerText.trim()) {
      setError('Please type your answer before submitting.')
      return
    }

    setSubmittingAnswer(true)
    setError(null)

    try {
      const response = await fetch(`${BACKEND_URL}/session/${session.session_id}/answer`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          question_id: currentQuestion.question_id,
          answer_text: answerText
        }),
      })

      if (!response.ok) {
        throw new Error('Failed to submit answer.')
      }

      const data = await response.json()
      setFeedback(data)
      setView('feedback')
    } catch (err) {
      console.error(err)
      setError(err.message)
    } finally {
      setSubmittingAnswer(false)
    }
  }

  // API Call: Fetch Summary
  const fetchSummary = async () => {
    if (!session) return
    
    try {
      const response = await fetch(`${BACKEND_URL}/session/${session.session_id}/summary`)
      if (!response.ok) {
        throw new Error('Failed to retrieve session summary.')
      }

      const data = await response.json()
      setSummary(data)
      setView('summary')
    } catch (err) {
      console.error(err)
      setError(err.message)
    }
  }

  // Reset to landing
  const handleReset = () => {
    setView('landing')
    setFile(null)
    setSession(null)
    setCurrentQuestion(null)
    setFeedback(null)
    setSummary(null)
    setAnswerText('')
    setError(null)
  }

  // Helper function to return signal icon & classes
  const getQualityBadge = (signal) => {
    switch(signal?.toLowerCase()) {
      case 'strong':
        return <span className="badge badge-strong"><CheckCircle size={14} /> Strong</span>
      case 'developing':
        return <span className="badge badge-developing"><Sparkles size={14} /> Developing</span>
      case 'needs_review':
        return <span className="badge badge-needs_review"><AlertCircle size={14} /> Needs Review</span>
      default:
        return <span className="badge">{signal}</span>
    }
  }

  return (
    <div className="app-container">
      {/* Header */}
      <header>
        <div className="logo-text" onClick={handleReset} style={{ cursor: 'pointer' }}>
          GroundedHire <span className="logo-dot"></span>
        </div>
        <div className="header-status">
          {session && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                Candidate: <strong>{session.candidate_name}</strong>
              </span>
              <span className="badge" style={{ background: 'rgba(255,255,255,0.05)', color: 'var(--primary)' }}>
                {role === 'ai_ml_engineer' ? 'AI/ML Engineer' : 'Data Scientist'}
              </span>
            </div>
          )}
        </div>
      </header>

      {/* Main Content Area */}
      <main>
        {/* Error Banner */}
        {error && (
          <div className="glass-panel animate-fade-in" style={{ 
            borderColor: 'var(--color-needs-review-border)', 
            background: 'var(--color-needs-review-bg)',
            padding: '12px 20px',
            marginBottom: '24px',
            borderRadius: '12px',
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            color: 'var(--color-needs-review)',
            fontSize: '0.9rem'
          }}>
            <AlertCircle size={18} />
            <span>{error}</span>
          </div>
        )}

        {/* 1. LANDING VIEW */}
        {view === 'landing' && (
          <div className="view-container animate-fade-in">
            <div style={{ textAlign: 'center', marginBottom: '40px' }}>
              <h1 style={{ fontSize: '3rem', marginBottom: '16px', fontWeight: '800' }}>
                Technical Interview <span className="gradient-text">Coaching</span>
              </h1>
              <p style={{ color: 'var(--text-secondary)', fontSize: '1.15rem', maxWidth: '600px', margin: '0 auto' }}>
                Prepare for senior machine learning roles with a grounded, conversational agent. Upload your resume to begin.
              </p>
            </div>

            <div className="glass-panel" style={{ padding: '32px', marginBottom: '32px' }}>
              {/* Role Selection */}
              <div className="form-group">
                <label className="form-label">1. Select Target Interview Role</label>
                <div className="role-cards">
                  <div 
                    className={`role-card ${role === 'ai_ml_engineer' ? 'selected' : ''}`}
                    onClick={() => setRole('ai_ml_engineer')}
                  >
                    <div className="role-card-title">AI / ML Engineer</div>
                    <div className="role-card-desc">Focuses on deployment, large-scale systems, deep learning optimizations, pipelines, and frameworks.</div>
                  </div>
                  <div 
                    className={`role-card ${role === 'data_science_applied_ml' ? 'selected' : ''}`}
                    onClick={() => setRole('data_science_applied_ml')}
                  >
                    <div className="role-card-title">Data Science & Applied ML</div>
                    <div className="role-card-desc">Focuses on statistical modeling, feature engineering, classification techniques, and metrics design.</div>
                  </div>
                </div>
              </div>

              {/* File Upload Zone */}
              <div className="form-group">
                <label className="form-label">2. Upload your Resume (PDF or TXT)</label>
                <div 
                  className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
                  onDragEnter={handleDrag}
                  onDragOver={handleDrag}
                  onDragLeave={handleDrag}
                  onDrop={handleDrop}
                  onClick={onButtonClick}
                >
                  <input 
                    ref={fileInputRef}
                    type="file" 
                    style={{ display: 'none' }} 
                    accept=".pdf,.txt"
                    onChange={handleFileChange}
                  />
                  <div className="upload-icon-container" style={{
                    width: '60px',
                    height: '60px',
                    borderRadius: '50%',
                    background: 'rgba(0, 242, 254, 0.05)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: 'var(--primary)',
                    boxShadow: '0 0 15px rgba(0, 242, 254, 0.05)'
                  }}>
                    {uploading ? (
                      <div className="spinner" style={{ width: '28px', height: '28px', borderWidth: '2px' }}></div>
                    ) : (
                      <Upload size={24} />
                    )}
                  </div>
                  {file ? (
                    <div>
                      <p style={{ fontWeight: '600', color: 'var(--primary)' }}>{file.name}</p>
                      <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>Click or drag to replace</p>
                    </div>
                  ) : (
                    <div>
                      <p style={{ fontWeight: '500' }}>Drag & drop your resume file here</p>
                      <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>or click to browse local files</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Optional Job Description & Target Company */}
              <div className="form-group" style={{ marginTop: '8px' }}>
                <div 
                  style={{
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'space-between',
                    cursor: 'pointer',
                    padding: '14px 18px',
                    borderRadius: '12px',
                    background: showJdInput ? 'rgba(0, 242, 254, 0.04)' : 'rgba(255, 255, 255, 0.02)',
                    border: showJdInput ? '1px solid rgba(0, 242, 254, 0.25)' : '1px solid var(--border-color)',
                    transition: 'all var(--transition-normal)'
                  }}
                  onClick={() => setShowJdInput(!showJdInput)}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                    <Briefcase size={18} style={{ color: 'var(--primary)' }} />
                    <span style={{ fontWeight: '600', color: 'var(--text-primary)' }}>
                      3. Target Company & Job Description (Optional)
                    </span>
                    {(company || jdText) && (
                      <span className="badge" style={{ background: 'rgba(0, 242, 254, 0.1)', color: 'var(--primary)', border: '1px solid rgba(0, 242, 254, 0.3)', padding: '2px 8px', fontSize: '0.7rem' }}>
                        Configured
                      </span>
                    )}
                  </div>
                  <span style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                    {showJdInput ? 'Collapse ▲' : 'Customize Mock Interview ▼'}
                  </span>
                </div>

                {showJdInput && (
                  <div className="animate-fade-in" style={{
                    marginTop: '12px',
                    padding: '20px',
                    borderRadius: '12px',
                    background: 'rgba(255, 255, 255, 0.01)',
                    border: '1px solid rgba(255, 255, 255, 0.06)'
                  }}>
                    {/* Target Company Name */}
                    <div style={{ marginBottom: '16px' }}>
                      <label className="form-label" style={{ fontSize: '0.85rem', marginBottom: '6px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <Building2 size={15} /> Target Company Name
                      </label>
                      <input 
                        type="text"
                        className="form-input"
                        placeholder="e.g. DeepMind, Google, Meta, Stripe, Startup"
                        value={company}
                        onChange={(e) => setCompany(e.target.value)}
                        style={{ width: '100%' }}
                      />
                    </div>

                    {/* Job Description Textarea & File Upload */}
                    <div>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                        <label className="form-label" style={{ fontSize: '0.85rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <FileText size={15} /> Job Description (JD)
                        </label>
                        <button
                          type="button"
                          className="btn btn-secondary"
                          style={{ padding: '4px 10px', fontSize: '0.75rem', height: 'auto' }}
                          onClick={() => jdFileInputRef.current && jdFileInputRef.current.click()}
                        >
                          <Upload size={12} /> Upload JD (.txt/.pdf)
                        </button>
                        <input 
                          ref={jdFileInputRef}
                          type="file" 
                          style={{ display: 'none' }} 
                          accept=".txt,.pdf"
                          onChange={handleJdFileChange}
                        />
                      </div>
                      <textarea
                        className="form-textarea"
                        rows={5}
                        placeholder="Paste the target job description here (responsibilities, required skills, preferred qualifications)..."
                        value={jdText}
                        onChange={(e) => setJdText(e.target.value)}
                        style={{ width: '100%', resize: 'vertical' }}
                      />
                      {jdFile && (
                        <div style={{ fontSize: '0.75rem', color: 'var(--primary)', marginTop: '4px' }}>
                          Loaded from file: {jdFile.name}
                        </div>
                      )}
                      <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                        When provided, interview topics will align directly with this job's requirements and seniority tier.
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Action Button */}
              <div style={{ textAlign: 'right', marginTop: '32px' }}>
                <button 
                  className={`btn btn-primary ${(!file || uploading) ? 'btn-disabled' : ''}`}
                  onClick={handleUploadAndStart}
                  disabled={!file || uploading}
                  style={{ width: '100%', justifyContent: 'center' }}
                >
                  {uploading ? 'Processing Resume...' : 'Analyze Resume & Plan Interview'}
                  <ChevronRight size={18} />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 2. READY TO START VIEW */}
        {view === 'ready' && session && (
          <div className="view-container animate-fade-in">
            <div className="glass-panel" style={{ padding: '40px', textAlign: 'center' }}>
              <div className="award-icon-container" style={{
                width: '80px',
                height: '80px',
                borderRadius: '50%',
                background: 'rgba(127, 0, 255, 0.1)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 24px',
                color: '#a855f7',
                boxShadow: '0 0 20px rgba(127, 0, 255, 0.1)'
              }}>
                <Award size={36} />
              </div>

              <h2 style={{ fontSize: '2rem', marginBottom: '16px' }}>
                Interview Prepared for <span className="gradient-text">{session.candidate_name}</span>
              </h2>

              <div style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: '8px',
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid var(--border-color)',
                padding: '8px 16px',
                borderRadius: '9999px',
                fontSize: '0.9rem',
                color: 'var(--text-secondary)',
                marginBottom: '32px'
              }}>
                <Layers size={16} style={{ color: 'var(--primary)' }} />
                <span>Estimated Experience Level: </span>
                <strong style={{ color: 'var(--text-primary)', textTransform: 'capitalize' }}>
                  {session.starting_difficulty_tier === 'intermediate-dense' ? 'advanced' : session.starting_difficulty_tier}
                </strong>
              </div>

              <p style={{
                background: 'rgba(255, 255, 255, 0.02)',
                borderLeft: '4px solid var(--primary)',
                padding: '24px',
                borderRadius: '0 12px 12px 0',
                textAlign: 'left',
                lineHeight: '1.6',
                fontSize: '1.05rem',
                color: 'var(--text-primary)',
                marginBottom: '40px'
              }}>
                "{session.greeting_message}"
              </p>

              <div style={{ display: 'flex', gap: '16px', justifyContent: 'center' }}>
                <button className="btn btn-secondary" onClick={handleReset} disabled={loadingQuestion}>
                  Back
                </button>
                <button 
                  className={`btn btn-primary ${loadingQuestion ? 'btn-disabled' : ''}`} 
                  onClick={fetchNextQuestion}
                  disabled={loadingQuestion}
                >
                  {loadingQuestion ? (
                    <>
                      <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }}></div>
                      Preparing Interview...
                    </>
                  ) : (
                    <>
                      Start Interview ({session.total_planned_questions} Questions)
                      <ChevronRight size={18} />
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 3. ACTIVE INTERVIEW VIEW */}
        {view === 'interview' && currentQuestion && (
          <div className="view-container animate-fade-in">
            {/* Progress indicator */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                QUESTION <strong>{currentQuestion.question_number}</strong> OF <strong>{currentQuestion.total_planned_questions}</strong>
              </span>
              <div style={{ display: 'flex', gap: '8px' }}>
                <span className="badge" style={{ background: 'rgba(0, 242, 254, 0.05)', color: 'var(--primary)' }}>
                  Topic: {currentQuestion.topic.replace(/_/g, ' ')}
                </span>
                <span className="badge" style={{ background: 'rgba(255,255,255,0.03)', color: 'var(--text-secondary)', border: '1px solid var(--border-color)' }}>
                  Difficulty: {currentQuestion.difficulty_tier === 'intermediate-dense' ? 'advanced' : currentQuestion.difficulty_tier}
                </span>
              </div>
            </div>

            {/* Progress Bar */}
            <div style={{ width: '100%', height: '4px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px', marginBottom: '32px', overflow: 'hidden' }}>
              <div style={{ 
                width: `${(currentQuestion.question_number / currentQuestion.total_planned_questions) * 100}%`, 
                height: '100%', 
                background: 'linear-gradient(90deg, var(--primary) 0%, #4facfe 100%)',
                transition: 'width var(--transition-normal)'
              }}></div>
            </div>

            {/* Question Card */}
            <div className="glass-panel" style={{ padding: '32px', marginBottom: '32px' }}>
              <h2 style={{ fontSize: '1.4rem', lineHeight: '1.5', fontWeight: '500', marginBottom: '24px', textAlign: 'left' }}>
                {currentQuestion.question_text}
              </h2>

              {/* Source Grounding Box (If present) */}
              {currentQuestion.source_book && (
                <div style={{
                  display: 'flex',
                  alignItems: 'flex-start',
                  gap: '12px',
                  background: 'rgba(0, 242, 254, 0.02)',
                  border: '1px dashed rgba(0, 242, 254, 0.15)',
                  padding: '16px',
                  borderRadius: '12px',
                  marginBottom: '32px',
                  textAlign: 'left'
                }}>
                  <BookOpen size={18} style={{ color: 'var(--primary)', marginTop: '2px', flexShrink: 0 }} />
                  <div>
                    <span style={{ fontSize: '0.75rem', fontWeight: '600', textTransform: 'uppercase', color: 'var(--primary)', letterSpacing: '0.05em', display: 'block', marginBottom: '4px' }}>
                      Coaching Reference Source
                    </span>
                    <span style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                      Matched with textbook <strong>{currentQuestion.source_book}</strong> in section <em>{currentQuestion.source_section_title}</em>
                    </span>
                  </div>
                </div>
              )}

              {/* Answer Input */}
              <div className="form-group">
                <label className="form-label">Your Response</label>
                <textarea 
                  className="form-textarea"
                  rows={7}
                  placeholder="Explain your approach, list details, or compare options clearly..."
                  value={answerText}
                  onChange={(e) => setAnswerText(e.target.value)}
                  disabled={submittingAnswer}
                />
              </div>

              {/* Actions */}
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <button 
                  className="btn btn-secondary"
                  onClick={() => setAnswerText('I would like to skip this question.')}
                  disabled={submittingAnswer}
                >
                  Skip
                </button>
                <button 
                  className={`btn btn-primary ${(submittingAnswer || !answerText.trim()) ? 'btn-disabled' : ''}`}
                  onClick={submitAnswer}
                  disabled={submittingAnswer || !answerText.trim()}
                >
                  {submittingAnswer ? (
                    <>
                      <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }}></div>
                      Evaluating response...
                    </>
                  ) : (
                    <>
                      Submit Answer
                      <ChevronRight size={18} />
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 4. FEEDBACK VIEW */}
        {view === 'feedback' && feedback && currentQuestion && (
          <div className="view-container animate-fade-in">
            {/* Status bar */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '24px' }}>
              <h3 style={{ fontSize: '1.25rem', fontFamily: 'var(--font-heading)' }}>
                Response Evaluation
              </h3>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                {getQualityBadge(feedback.quality_signal)}
                {feedback.current_difficulty_tier !== currentQuestion.difficulty_tier && (
                  <span className="badge" style={{ 
                    background: 'rgba(255, 255, 255, 0.03)', 
                    color: 'var(--text-secondary)',
                    border: '1px solid var(--border-color)',
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}>
                    {feedback.current_difficulty_tier === 'intermediate-dense' && currentQuestion.difficulty_tier === 'intermediate' ? (
                      <><TrendingUp size={12} style={{ color: 'var(--color-strong)' }} /> Harder Question next</>
                    ) : feedback.current_difficulty_tier === 'intermediate' && currentQuestion.difficulty_tier === 'intermediate-dense' ? (
                      <><TrendingDown size={12} style={{ color: 'var(--color-needs-review)' }} /> Simpler Question next</>
                    ) : feedback.current_difficulty_tier === 'intermediate' && currentQuestion.difficulty_tier === 'beginner' ? (
                      <><TrendingUp size={12} style={{ color: 'var(--color-strong)' }} /> Harder Question next</>
                    ) : feedback.current_difficulty_tier === 'beginner' && currentQuestion.difficulty_tier === 'intermediate' ? (
                      <><TrendingDown size={12} style={{ color: 'var(--color-needs-review)' }} /> Simpler Question next</>
                    ) : (
                      <><Activity size={12} /> Difficulty tier updated</>
                    )}
                  </span>
                )}
              </div>
            </div>

            {/* Main Coach Card */}
            <div className="glass-panel" style={{ padding: '32px', marginBottom: '32px' }}>
              {/* Question overview */}
              <div style={{ marginBottom: '20px', borderBottom: '1px solid rgba(255, 255, 255, 0.05)', paddingBottom: '20px' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Question asked</span>
                <p style={{ fontSize: '1.1rem', fontWeight: '500', marginTop: '4px' }}>{currentQuestion.question_text}</p>
              </div>

              {/* Coaching Feedback text */}
              <div style={{ marginBottom: '24px' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Coaching Feedback</span>
                <p style={{ 
                  fontSize: '1.05rem', 
                  marginTop: '6px', 
                  color: 'var(--text-primary)', 
                  background: 'rgba(255,255,255,0.01)',
                  padding: '16px',
                  borderRadius: '8px',
                  border: '1px solid rgba(255, 255, 255, 0.03)'
                }}>{feedback.brief_feedback}</p>
              </div>



              {/* Actions */}
              <div style={{ textAlign: 'right' }}>
                <button 
                  className={`btn btn-primary ${loadingQuestion ? 'btn-disabled' : ''}`} 
                  onClick={fetchNextQuestion}
                  disabled={loadingQuestion}
                >
                  {loadingQuestion ? (
                    <>
                      <div className="spinner" style={{ width: '16px', height: '16px', borderWidth: '2px' }}></div>
                      Retrieving next question...
                    </>
                  ) : (
                    <>
                      {currentQuestion.question_number >= currentQuestion.total_planned_questions ? 'Finish & Generate Report' : 'Next Question'}
                      <ChevronRight size={18} />
                    </>
                  )}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 5. SUMMARY REPORT VIEW */}
        {view === 'summary' && summary && (
          <div className="view-container animate-fade-in">
            {/* Header info */}
            <div style={{ textAlign: 'center', marginBottom: '40px' }}>
              <div className="award-icon-container" style={{
                width: '64px',
                height: '64px',
                borderRadius: '50%',
                background: 'rgba(0, 242, 254, 0.08)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                margin: '0 auto 16px',
                color: 'var(--primary)',
                boxShadow: '0 0 20px rgba(0, 242, 254, 0.1)'
              }}>
                <ClipboardList size={30} />
              </div>
              <h2 style={{ fontSize: '2rem', marginBottom: '8px' }}>Session Performance Report</h2>
              <p style={{ color: 'var(--text-secondary)' }}>
                Candidate: <strong>{summary.candidate_name}</strong> | Role: <strong>{summary.role === 'ai_ml_engineer' ? 'AI/ML Engineer' : 'Data Scientist'}</strong>
              </p>
            </div>

            {/* Overall impression card */}
            <div className="glass-panel" style={{ padding: '32px', marginBottom: '32px', borderLeft: '5px solid var(--primary)' }}>
              <h3 style={{ fontSize: '1.2rem', marginBottom: '12px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Sparkles size={18} style={{ color: 'var(--primary)' }} />
                Overall Coach Impression
              </h3>
              <p style={{ fontSize: '1.05rem', lineHeight: '1.6', color: 'var(--text-primary)' }}>
                "{summary.overall_impression}"
              </p>
            </div>

            {/* Statistics and quality distribution grid */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '24px', marginBottom: '32px' }}>
              {/* Count block */}
              <div className="glass-panel" style={{ padding: '24px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
                <span style={{ fontSize: '3.5rem', fontWeight: '800', color: 'var(--primary)', lineHeight: '1' }}>
                  {summary.qa_pairs ? summary.qa_pairs.length : 0}
                </span>
                <span style={{ fontSize: '0.8rem', fontWeight: '600', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', marginTop: '8px' }}>
                  Questions Answered
                </span>
              </div>

              {/* Progress bars block */}
              <div className="glass-panel" style={{ padding: '24px' }}>
                <h4 style={{ fontSize: '0.9rem', fontWeight: '700', textTransform: 'uppercase', color: 'var(--text-secondary)', letterSpacing: '0.05em', marginBottom: '16px' }}>
                  Quality Evaluation Distribution
                </h4>
                
                {/* Strong Bar */}
                <div style={{ marginBottom: '14px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                    <span style={{ color: 'var(--color-strong)', fontWeight: '600' }}>Strong Answers</span>
                    <span>{summary.quality_distribution?.strong || 0}</span>
                  </div>
                  <div style={{ height: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ 
                      width: `${(summary.quality_distribution?.strong / summary.qa_pairs?.length) * 100 || 0}%`, 
                      height: '100%', 
                      background: 'var(--color-strong)' 
                    }}></div>
                  </div>
                </div>

                {/* Developing Bar */}
                <div style={{ marginBottom: '14px' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                    <span style={{ color: 'var(--color-developing)', fontWeight: '600' }}>Developing Answers</span>
                    <span>{summary.quality_distribution?.developing || 0}</span>
                  </div>
                  <div style={{ height: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ 
                      width: `${(summary.quality_distribution?.developing / summary.qa_pairs?.length) * 100 || 0}%`, 
                      height: '100%', 
                      background: 'var(--color-developing)' 
                    }}></div>
                  </div>
                </div>

                {/* Needs Review Bar */}
                <div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '4px' }}>
                    <span style={{ color: 'var(--color-needs-review)', fontWeight: '600' }}>Needs Review</span>
                    <span>{summary.quality_distribution?.needs_review || 0}</span>
                  </div>
                  <div style={{ height: '8px', background: 'rgba(255,255,255,0.03)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ 
                      width: `${(summary.quality_distribution?.needs_review / summary.qa_pairs?.length) * 100 || 0}%`, 
                      height: '100%', 
                      background: 'var(--color-needs-review)' 
                    }}></div>
                  </div>
                </div>

              </div>
            </div>

            {/* Q&A Detailed History */}
            <h3 style={{ fontSize: '1.25rem', marginBottom: '16px', fontFamily: 'var(--font-heading)' }}>
              Interview Question History
            </h3>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '16px', marginBottom: '40px' }}>
              {summary.qa_pairs && summary.qa_pairs.map((pair, idx) => (
                <div key={idx} className="glass-panel" style={{ padding: '24px', textAlign: 'left' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '12px', marginBottom: '12px' }}>
                    <span style={{ fontSize: '0.8rem', fontWeight: '700', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                      Question {idx + 1} | {pair.question_type}
                    </span>
                    {getQualityBadge(pair.quality_signal)}
                  </div>
                  <h4 style={{ fontSize: '1.1rem', fontWeight: '600', marginBottom: '16px', color: 'var(--text-primary)' }}>
                    {pair.question_text}
                  </h4>
                  <div style={{ 
                    background: 'rgba(0,0,0,0.2)', 
                    padding: '16px', 
                    borderRadius: '8px', 
                    border: '1px solid var(--border-color)',
                    fontSize: '0.9rem',
                    marginBottom: '12px'
                  }}>
                    <strong style={{ color: 'var(--text-secondary)', display: 'block', marginBottom: '4px' }}>Candidate Answer:</strong>
                    <p style={{ color: 'var(--text-primary)', whiteSpace: 'pre-wrap' }}>{pair.answer_text || 'No answer provided.'}</p>
                  </div>
                  <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)', display: 'flex', gap: '6px' }}>
                    <strong style={{ color: 'var(--primary)' }}>Feedback:</strong>
                    <span>{pair.brief_feedback}</span>
                  </div>
                </div>
              ))}
            </div>

            {/* Action buttons */}
            <div style={{ textAlign: 'center', marginBottom: '40px' }}>
              <button className="btn btn-primary" onClick={handleReset}>
                <RefreshCw size={18} />
                Restart New Interview
              </button>
            </div>

          </div>
        )}
      </main>

      {/* Footer */}
      <footer>
        &copy; {new Date().getFullYear()} GroundedHire Adaptive Coaching Platform. Powered by Retrieval-Augmented Generation.
      </footer>
    </div>
  )
}

export default App
