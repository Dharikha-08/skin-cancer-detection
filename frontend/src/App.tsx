import { useState, useRef } from 'react'
import './App.css'

interface PredictionResult {
  label: string;
  probability: number;
  breakdown: {
    image_component: number;
    tabular_component: number;
  };
  fusion_weight: number;
  threshold: number;
}

function App() {
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Tabular Form State
  const [formData, setFormData] = useState({
    sex: 'male',
    location: 'back',
    elevation: 'palpable',
    diff: 'medium',
    score: 0,
    pig_net: 'absent',
    streaks: 'absent',
    pigment: 'absent',
    reg_struc: 'absent',
    dots: 'absent',
    blue_veil: 'absent',
    vasc: 'absent'
  });

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleInputChange = (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement> | File) => {
    let selectedFile: File | undefined;
    if (e instanceof File) selectedFile = e;
    else if (e.target.files && e.target.files[0]) selectedFile = e.target.files[0];

    if (!selectedFile) return;
    setError(null);
    const reader = new FileReader();
    reader.onloadend = () => setPreview(reader.result as string);
    reader.readAsDataURL(selectedFile);
    setResult(null);
  };

  const onDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(true);
  };

  const onDragLeave = () => {
    setDragActive(false);
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!preview) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const response = await fetch(preview);
      const blob = await response.blob();

      const dataLayer = new FormData();
      dataLayer.append('file', blob, 'analysis.jpg');
      
      // Append all tabular data
      Object.entries(formData).forEach(([key, value]) => {
        dataLayer.append(key, value.toString());
      });

      const res = await fetch('https://skin-cancer-detection-uthu.onrender.com/predict', {
        method: 'POST',
        body: dataLayer,
      });

      if (!res.ok) throw new Error('Analytical server error');
      const data = await res.json();
      if (data.status === "error") throw new Error(data.message);
      setResult(data);
    } catch (err: any) {
      setError(err.message || "Detection failed.");
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setPreview(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  return (
    <div className="container">
      <header>
        <h1 className="hero-title">DermVision AI 4.0</h1>
        <p className="hero-subtitle">Optimized Multimodal Skin Analysis System</p>
      </header>

      <main className="multimodal-grid">
        <section className="input-section">
          {/* Clinical Metadata Form */}
          <div className="form-card">
            <h3>Clinical Metadata</h3>
            <div className="form-grid">
              <div className="input-group">
                <label>Sex</label>
                <select name="sex" value={formData.sex} onChange={handleInputChange}>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </div>
              <div className="input-group">
                <label>Location</label>
                <select name="location" value={formData.location} onChange={handleInputChange}>
                  <option value="back">Back</option>
                  <option value="lower limbs">Lower Limbs</option>
                  <option value="upper limbs">Upper Limbs</option>
                  <option value="head neck">Head/Neck</option>
                  <option value="abdomen">Abdomen</option>
                  <option value="chest">Chest</option>
                </select>
              </div>
              <div className="input-group">
                <label>Elevation</label>
                <select name="elevation" value={formData.elevation} onChange={handleInputChange}>
                  <option value="flat">Flat</option>
                  <option value="palpable">Palpable</option>
                  <option value="nodular">Nodular</option>
                </select>
              </div>
              <div className="input-group">
                <label>Difficulty</label>
                <select name="diff" value={formData.diff} onChange={handleInputChange}>
                  <option value="low">Low</option>
                  <option value="medium">Medium</option>
                  <option value="high">High</option>
                </select>
              </div>
              <div className="input-group full-width">
                <label>Lesion Characteristics (Dermoscopy)</label>
                <div className="checkbox-flex">
                   <select name="pig_net" value={formData.pig_net} onChange={handleInputChange}>
                      <option value="absent">Pigment Network: Absent</option>
                      <option value="typical">Pigment Network: Typical</option>
                      <option value="atypical">Pigment Network: Atypical</option>
                   </select>
                   <select name="streaks" value={formData.streaks} onChange={handleInputChange}>
                      <option value="absent">Streaks: Absent</option>
                      <option value="regular">Streaks: Regular</option>
                      <option value="irregular">Streaks: Irregular</option>
                   </select>
                   <select name="pigment" value={formData.pigment} onChange={handleInputChange}>
                      <option value="absent">Pigmentation: Absent</option>
                      <option value="diffuse regular">Pigment: Diffuse Regular</option>
                      <option value="diffuse irregular">Pigment: Diffuse Irregular</option>
                   </select>
                </div>
              </div>
            </div>
          </div>

          {/* Image Upload Zone */}
          <div 
            className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
            onClick={() => fileInputRef.current?.click()}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
          >
            <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="image/*" style={{ display: 'none' }} />
            {preview ? (
              <div className="preview-container">
                <img src={preview} alt="Lesion" className="preview-img" />
                <button className="btn-clear" onClick={(e) => { e.stopPropagation(); clear(); }}>Change Image</button>
              </div>
            ) : (
              <div className="upload-cta">
                <svg width="48" height="48" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" /></svg>
                <p>Upload Clinical Image</p>
              </div>
            )}
          </div>

          <button className="btn-primary" onClick={handleUpload} disabled={loading || !preview}>
             {loading ? <div className="loader tiny"></div> : "RUN MULTIMODAL ANALYSIS"}
          </button>
        </section>

        <section className="results-section">
          {result ? (
            <div className={`result-card ${result.label.toLowerCase()}`}>
              <div className="result-header">
                <h2>{result.label === 'MALIGNANT' ? '🚨 MALIGNANT' : '✅ BENIGN'}</h2>
                <div className="prob-badge">{Math.round(result.probability * 100)}% Risk Score</div>
              </div>

              <div className="fusion-breakdown">
                <h4>Analysis Breakdown (Late Fusion)</h4>
                <div className="breakdown-row">
                   <span>Image Model Confidence:</span>
                   <span>{Math.round(result.breakdown.image_component * 100)}%</span>
                </div>
                <div className="breakdown-row">
                   <span>Tabular Model Confidence:</span>
                   <span>{Math.round(result.breakdown.tabular_component * 100)}%</span>
                </div>
                <div className="fusion-footer">
                   Used Fusion Weight (α): {result.fusion_weight}
                </div>
              </div>

              <div className="progress-track">
                <div className="progress-fill" style={{ width: `${result.probability * 100}%` }}></div>
              </div>

              <p className="medical-disclaimer">
                Validated analysis using EfficientNet-B2 (Image) and XGBoost (Tabular). 
                Threshold used: {result.threshold}.
              </p>
              <button className="btn-secondary" onClick={clear}>New Patient</button>
            </div>
          ) : (
            <div className="empty-results">
               <p>Results will be displayed here after analysis.</p>
            </div>
          )}
          {error && <div className="error-box">{error}</div>}
        </section>
      </main>
    </div>
  );
}

export default App;
