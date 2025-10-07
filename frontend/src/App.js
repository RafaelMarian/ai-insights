// frontend/src/App.js
import React, { useEffect, useState } from "react";
import "./App.css";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

const API = process.env.REACT_APP_API_URL || "http://localhost:5000";

function App() {
  const [regions, setRegions] = useState([]);
  const [productName, setProductName] = useState("");
  const [productType, setProductType] = useState("Cold Medicine");
  const [targetAge, setTargetAge] = useState("Seniors");
  const [averagePrice, setAveragePrice] = useState(20);
  const [season, setSeason] = useState("Winter");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    // load regions
    fetch(`${API}/regions`)
      .then((r) => r.json())
      .then((data) => setRegions(data))
      .catch((err) => console.error("Could not load regions:", err));
  }, []);

  const handlePredict = async () => {
    setLoading(true);
    setResult(null);
    try {
      const resp = await fetch(`${API}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          product_name: productName,
          product_type: productType,
          target_age_group: targetAge,
          average_price: averagePrice,
          region: regions[0] || "Criseni",
          season
        })
      });

      if (!resp.ok) {
        const txt = await resp.text();
        throw new Error(`${resp.status} ${txt}`);
      }
      const data = await resp.json();
      setResult(data);
    } catch (err) {
      setResult({ error: err.message });
    } finally {
      setLoading(false);
    }
  };

  const renderImportanceChart = (fi) => {
    if (!fi || Object.keys(fi).length === 0) return null;
    const data = Object.entries(fi).map(([k, v]) => ({ name: k, value: Math.round(v * 100) }));
    return (
      <div style={{ width: "100%", height: 250 }}>
        <ResponsiveContainer>
          <BarChart data={data}>
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="value" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  };

  return (
    <div className="App" style={{ maxWidth: 980, margin: "40px auto", padding: 20 }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <h1>AI Pharma Market Predictor</h1>
        <small>Predict product success by region (Demo)</small>
      </header>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginTop: 20 }}>
        <div style={{ padding: 16, borderRadius: 8, boxShadow: "0 2px 8px rgba(0,0,0,0.06)" }}>
          <h3>Product info</h3>
          <input placeholder="Product name" value={productName} onChange={(e) => setProductName(e.target.value)} style={{ width: "100%", padding: 8, marginBottom: 8 }} />

          <label>Type</label>
          <select value={productType} onChange={(e) => setProductType(e.target.value)} style={{ width: "100%", padding: 8, marginBottom: 8 }}>
            <option>Cold Medicine</option>
            <option>Vitamins</option>
            <option>Painkiller</option>
            <option>Allergy</option>
          </select>

          <label>Target age group</label>
          <select value={targetAge} onChange={(e) => setTargetAge(e.target.value)} style={{ width: "100%", padding: 8, marginBottom: 8 }}>
            <option>Children</option>
            <option>Adults</option>
            <option>Seniors</option>
          </select>

          <label>Average price (RON)</label>
          <input type="number" value={averagePrice} onChange={(e) => setAveragePrice(e.target.value)} style={{ width: "100%", padding: 8 }} />
        </div>

        <div style={{ padding: 16, borderRadius: 8, boxShadow: "0 2px 8px rgba(0,0,0,0.06)" }}>
          <h3>Region & context</h3>
          <label>Region</label>
          <select style={{ width: "100%", padding: 8, marginBottom: 8 }} onChange={(e) => { /* keep selection in body */ }}>
            {regions.length === 0 ? <option>Loading...</option> : regions.map((r) => <option key={r}>{r}</option>)}
          </select>

          <label>Season</label>
          <select value={season} onChange={(e) => setSeason(e.target.value)} style={{ width: "100%", padding: 8 }}>
            <option>Winter</option>
            <option>Spring</option>
            <option>Summer</option>
            <option>Autumn</option>
          </select>

          <div style={{ marginTop: 16 }}>
            <button onClick={handlePredict} disabled={loading} style={{ padding: "10px 18px", fontSize: 16 }}>
              {loading ? "Predicting..." : "Predict Market Success"}
            </button>
          </div>
        </div>
      </div>

      <div style={{ marginTop: 24 }}>
        {result ? (
          result.error ? (
            <div style={{ color: "red" }}>Error: {result.error}</div>
          ) : (
            <div style={{ padding: 16, borderRadius: 8, boxShadow: "0 2px 10px rgba(0,0,0,0.06)", background: "#fff" }}>
              <h2>{result.Product}</h2>
              <h3>Predicted success: <span style={{ color: "#1f8a3e" }}>{result.Predicted_Success}</span></h3>
              <p>{result.Interpretation}</p>

              <h4>Feature importance</h4>
              {renderImportanceChart(result.Feature_Importance)}
            </div>
          )
        ) : (
          <div style={{ color: "#666" }}>Enter product & region then press Predict.</div>
        )}
      </div>
    </div>
  );
}

export default App;
