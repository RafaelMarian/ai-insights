import React, { useState } from "react";
import "./App.css";

function App() {
  const [age, setAge] = useState("");
  const [income, setIncome] = useState("");
  const [result, setResult] = useState("");

  const handlePredict = async () => {
    try {
      const response = await fetch("${process.env.REACT_APP_API_URL}/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ Age: parseInt(age), Income: parseInt(income) })
      });

      const data = await response.json();
      if (data.Prediction) {
        setResult(data.Prediction);
      } else {
        setResult("Error: " + (data.error || "Unknown error"));
      }
    } catch (err) {
      setResult("Error: " + err.message);
    }
  };

  return (
    <div className="App" style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>AI Insights Predictor</h1>
      <div style={{ margin: "20px" }}>
        <input
          type="number"
          placeholder="Age"
          value={age}
          onChange={(e) => setAge(e.target.value)}
        />
      </div>
      <div style={{ margin: "20px" }}>
        <input
          type="number"
          placeholder="Income"
          value={income}
          onChange={(e) => setIncome(e.target.value)}
        />
      </div>
      <button onClick={handlePredict}>Predict</button>
      <h2 style={{ marginTop: "20px" }}>{result}</h2>
    </div>
  );
}

export default App;
