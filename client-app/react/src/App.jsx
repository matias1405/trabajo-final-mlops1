import { useEffect, useState } from "react";
import "./App.css";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

const initialForm = {
  budget: "10000000",
  runtime: "120",
  original_language: "en",
  release_date: "2024-07-15",
  genres: "Action,Comedy",
  production_countries: "United States of America",
  production_companies: "Warner Bros.",
};

function parseList(value) {
  return value
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function App() {
  const [health, setHealth] = useState(null);
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((res) => res.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "unreachable" }));
  }, []);

  async function handleSubmit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    const payload = {
      runtime: Number(form.runtime),
      budget: Number(form.budget),
      original_language: form.original_language.trim(),
      release_date: form.release_date,
      genres: parseList(form.genres),
      production_countries: parseList(form.production_countries),
      production_companies: parseList(form.production_companies),
    };

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }

      const data = await response.json();
      setResult(data);
    } catch {
      setError("No se pudo conectar con el backend o el modelo falló al predecir.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="hero">
        <div className="hero__copy">
          <p className="eyebrow">Prediction Movies</p>
          <h1>Probá el modelo de rentabilidad de películas.</h1>
          <p className="lede">
            Enviá datos de una película y obtené una predicción del modelo
            servido por FastAPI.
          </p>

          <div className="status-card">
            <span className="status-label">Backend</span>
            <strong>{API_URL}</strong>
            <pre>{health ? JSON.stringify(health, null, 2) : "cargando..."}</pre>
          </div>
        </div>

        <form className="panel" onSubmit={handleSubmit}>
          <div className="grid">
            <label>
              Budget
              <input
                type="number"
                value={form.budget}
                onChange={(event) =>
                  setForm({ ...form, budget: event.target.value })
                }
              />
            </label>
            <label>
              Runtime
              <input
                type="number"
                value={form.runtime}
                onChange={(event) =>
                  setForm({ ...form, runtime: event.target.value })
                }
              />
            </label>
          </div>

          <label>
            Original language
            <input
              type="text"
              value={form.original_language}
              onChange={(event) =>
                setForm({ ...form, original_language: event.target.value })
              }
            />
          </label>
          <label>
            Release date
            <input
              type="date"
              value={form.release_date}
              onChange={(event) =>
                setForm({ ...form, release_date: event.target.value })
              }
            />
          </label>
          <label>
            Genres
            <textarea
              rows="2"
              value={form.genres}
              onChange={(event) => setForm({ ...form, genres: event.target.value })}
            />
          </label>

          <label>
            Production countries
            <textarea
              rows="2"
              value={form.production_countries}
              onChange={(event) =>
                setForm({ ...form, production_countries: event.target.value })
              }
            />
          </label>

          <label>
            Production companies
            <textarea
              rows="2"
              value={form.production_companies}
              onChange={(event) =>
                setForm({ ...form, production_companies: event.target.value })
              }
            />
          </label>

          <button type="submit" disabled={loading}>
            {loading ? "Prediciendo..." : "Predecir"}
          </button>

          {error ? <div className="feedback feedback--error">{error}</div> : null}
          {result ? (
            <div className="feedback feedback--success">
              <div>
                <span>Predicción</span>
                <strong>{String(result.prediction)}</strong>
              </div>
              {typeof result.probability === "number" ? (
                <div>
                  <span>Probability</span>
                  <strong>{result.probability.toFixed(2)}</strong>
                </div>
              ) : null}
            </div>
          ) : null}
        </form>
      </section>
    </main>
  );
}

export default App;
