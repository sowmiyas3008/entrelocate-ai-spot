
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";
import "leaflet/dist/leaflet.css";
import "@/utils/fixLeafletIcons";



ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);



// import React from "react";
// import ReactDOM from "react-dom/client";

// const App = () => {
//   return (
//     <div style={{ color: "red", fontSize: "40px" }}>
//       APP IS RENDERING
//     </div>
//   );
// };

// const rootElement = document.getElementById("root");

// if (!rootElement) {
//   console.error("ROOT ELEMENT NOT FOUND");
// } else {
//   ReactDOM.createRoot(rootElement).render(
//     <React.StrictMode>
//       <App />
//     </React.StrictMode>
//   );
// }
