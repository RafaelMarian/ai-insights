import React, { useEffect, useState } from "react";
import "./App.css";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip as ReTooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";

const API = process.env.REACT_APP_API_URL || "http://localhost:5000";

function Spinner() {
  return <div className="spinner" aria-hidden="true"></div>;
}

function formatPercentStr(s) {
  // keep original formatting if already a percent string
  if (!s) return "";
  if (typeof s === "string" && s.includes("%")) return s;
  return `${(s * 100).toFixed(1)}%`;
}

function App() {
  const [regions, setRegions] = useState([]);
  const [selectedRegion, setSelectedRegion] = useState("");
  const [productName, setProductName] = useState("");
  const [productType, setProductType] = useState("Cold Medicine");
  const [targetAge, setTargetAge] = useState("Seniors");
  const [averagePrice, setAveragePrice] = useState(20);
  const [season, setSeason] = useState("Winter");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    fetch(`${API}/api/regions`)
      .then((r) => {
        if (!r.ok) throw new Error(`Status ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setRegions(data);
        if (data.length > 0) setSelectedRegion(data[0]);
      })
      .catch((err) => {
        console.error("Could not load regions:", err);
        setErrorMsg("Could not load regions from the server.");
      });
  }, []);

  const resetForm = () => {
    setProductName("");
    setProductType("Cold Medicine");
    setTargetAge("Seniors");
    setAveragePrice(20);
    setSeason("Winter");
    setSelectedRegion(regions[0] || "");
    setResult(null);
    setErrorMsg("");
  };

  const handlePredict = async () => {
    setLoading(true);
    setResult(null);
    setErrorMsg("");
    try {
      const resp = await fetch(`${API}/api/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          product_name: productName || "Unnamed Product",
          product_type: productType,
          target_age_group: targetAge,
          average_price: Number(averagePrice),
          region: selectedRegion,
          season,
        }),
      });

      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`${resp.status} ${txt}`);
      }
      const data = await resp.json();
      setResult(data);
    } catch (err) {
      console.error("Prediction error:", err);
      setErrorMsg(err.message || "Unknown error");
    } finally {
      setLoading(false);
    }
  };

  const renderImportanceChart = (fi) => {
    if (!fi || Object.keys(fi).length === 0) return null;
    const data = Object.entries(fi).map(([k, v]) => ({
      name: k.replace(/_/g, " "),
      value: Math.round(v * 100),
      raw: v,
    }));
    const colors = ["#4f46e5", "#06b6d4", "#f97316", "#10b981", "#ef4444", "#8b5cf6"];
    return (
      <div className="card chart-card">
        <h4 className="card-title">Feature importance</h4>
        <div style={{ width: "100%", height: 260 }}>
          <ResponsiveContainer>
            <BarChart data={data} layout="vertical" margin={{ left: 40, right: 20 }}>
              <XAxis type="number" hide />
              <YAxis width={140} dataKey="name" type="category" />
              <ReTooltip formatter={(value, name, props) => [`${value}%`, props.payload.name]} />
              <Bar dataKey="value" barSize={18}>
                {data.map((entry, idx) => (
                  <Cell key={`cell-${idx}`} fill={colors[idx % colors.length]} title={`${entry.name}: ${entry.value}%`} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    );
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <h1>AI Pharma Market Predictor</h1>
          <p className="muted">Estimate product success by region — demo</p>
        </div>
      </header>

      <main className="container">
        <section className="grid">
          <div className="card">
            <h3 className="card-title">Product information</h3>

            <label className="label">Product name</label>
            <input
              className="input"
              placeholder="Ex: Antigripin"
              value={productName}
              onChange={(e) => setProductName(e.target.value)}
            />

            <div className="row">
              <div style={{ flex: 1 }}>
                <label className="label">Type</label>
                <select className="select" value={productType} onChange={(e) => setProductType(e.target.value)}>
                  <option>Cold Medicine</option>
                  <option>Vitamins</option>
                  <option>Painkiller</option>
                  <option>Allergy</option>
                </select>
              </div>

              <div style={{ flex: 1, marginLeft: 12 }}>
                <label className="label">Target age group</label>
                <select className="select" value={targetAge} onChange={(e) => setTargetAge(e.target.value)}>
                  <option>Children</option>
                  <option>Adults</option>
                  <option>Seniors</option>
                </select>
              </div>
            </div>

            <label className="label">Average price (RON)</label>
            <input className="input" type="number" value={averagePrice} onChange={(e) => setAveragePrice(e.target.value)} />
          </div>

          <div className="card">
            <h3 className="card-title">Region & context</h3>

            <label className="label">Region</label>
            <select className="select" value={selectedRegion} onChange={(e) => setSelectedRegion(e.target.value)}>
              {regions.length === 0 ? <option>Loading...</option> : regions.map((r) => <option key={r}>{r}</option>)}
            </select>

            <label className="label">Season</label>
            <select className="select" value={season} onChange={(e) => setSeason(e.target.value)}>
              <option>Winter</option>
              <option>Spring</option>
              <option>Summer</option>
              <option>Autumn</option>
            </select>

            <div className="actions">
              <button className="btn primary" onClick={handlePredict} disabled={loading || !selectedRegion}>
                {loading ? (
                  <>
                    <Spinner /> Predicting...
                  </>
                ) : (
                  "Predict Market Success"
                )}
              </button>

              <button className="btn ghost" onClick={resetForm} disabled={loading}>
                Reset
              </button>
            </div>

            {errorMsg && <div className="error">{errorMsg}</div>}
          </div>
        </section>

        <section className="results">
          {result ? (
            result.error ? (
              <div className="card">
                <h3 className="card-title">Error</h3>
                <p className="error">{result.error}</p>
              </div>
            ) : (
              <>
                <div className="card result-card">
                  <div className="result-left">
                    <h2 className="product-name">{result.Product}</h2>
                    <p className="muted">Prediction: <strong className="big">{result.Predicted_Success}</strong></p>
                    <p className="muted">Label: <strong>{result.PredictionLabel}</strong></p>
                    <p className="muted small">{result.Interpretation}</p>
                  </div>
                  <div className="result-right">
                    <div className="score-circle" title={`Probability: ${result.Predicted_Success}`}>
                      <div className="score-text">{result.Predicted_Success}</div>
                    </div>
                  </div>
                </div>

                {renderImportanceChart(result.Feature_Importance)}
              </>
            )
          ) : (
            <div className="card placeholder">
              <p className="muted">Complete the form and press Predict to see results here.</p>
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        <small>Demo • AI Pharma Market Predictor • Built for portfolio</small>
      </footer>
    </div>
  );
}

export default App;
