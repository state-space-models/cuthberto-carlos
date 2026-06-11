import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "flag-icons/css/flag-icons.min.css";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
