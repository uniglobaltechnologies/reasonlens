import { BrowserRouter, Routes, Route } from "react-router-dom";
import Hub from "./pages/Hub";
import Audit from "./pages/Audit";
import AuditRuns from "./pages/AuditRuns";
import Evaluate from "./pages/Evaluate";
import NotFound from "./pages/NotFound";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Hub />} />
        <Route path="/audit" element={<Audit />} />
        <Route path="/audit/runs" element={<AuditRuns />} />
        {/* <Route path="/audit/runs/:id" element={<AuditDetail />} /> */}
        <Route path="/evaluate" element={<Evaluate />} />
        {/* <Route path="/assess" element={<Assess />} /> */}
        {/* <Route path="/frameworks" element={<Frameworks />} /> */}
        {/* <Route path="/policy" element={<Policy />} /> */}
        {/* <Route path="/my-progress" element={<MyProgress />} /> */}
        {/* <Route path="/portfolio" element={<Portfolio />} /> */}
        {/* <Route path="/badges" element={<Badges />} /> */}
        {/* <Route path="/auth" element={<Auth />} /> */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
