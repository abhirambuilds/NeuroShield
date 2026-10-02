import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import Dataset from "./pages/Dataset";
import Models from "./pages/Models";
import Training from "./pages/Training";
import Prediction from "./pages/Prediction";
import Uncertainty from "./pages/Uncertainty";
import Evaluation from "./pages/Evaluation";
import About from "./pages/About";

export default function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/dataset" element={<Dataset />} />
        <Route path="/models" element={<Models />} />
        <Route path="/training" element={<Training />} />
        <Route path="/prediction" element={<Prediction />} />
        <Route path="/uncertainty" element={<Uncertainty />} />
        <Route path="/evaluation" element={<Evaluation />} />
        <Route path="/about" element={<About />} />
      </Routes>
    </Layout>
  );
}
