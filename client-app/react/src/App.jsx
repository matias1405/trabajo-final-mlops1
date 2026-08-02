import { useEffect, useState } from "react";

// La URL del backend FastAPI se fija en build time (Vite), permitiendo que
// la misma app se reconstruya para Staging o Production apuntando a cada API.
const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function App() {
  const [status, setStatus] = useState(null);

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((res) => res.json())
      .then(setStatus)
      .catch(() => setStatus({ status: "unreachable" }));
  }, []);

  return (
    <div>
      <h1>Prediction Movies</h1>
      <p>Backend: {API_URL}</p>
      <p>Status: {status ? JSON.stringify(status) : "cargando..."}</p>
    </div>
  );
}

export default App;
