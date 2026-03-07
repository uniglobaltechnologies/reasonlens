import { BrowserRouter, Routes, Route } from "react-router-dom";
import Hub from "./pages/Hub";
import Audit from "./pages/Audit";
import AuditRuns from "./pages/AuditRuns";
import AuditDetail from "./pages/AuditDetail";
import Evaluate from "./pages/Evaluate";
import Frameworks from "./pages/Frameworks";
import FrameworkDetail from "./pages/FrameworkDetail";
import Assess from "./pages/Assess";
import Policy from "./pages/Policy";
import MyProgress from "./pages/MyProgress";
import Portfolio from "./pages/Portfolio";
import Badges from "./pages/Badges";
import Auth from "./pages/Auth";
import NotFound from "./pages/NotFound";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Hub />} />
        {/* Audit */}
        <Route path="/audit" element={<Audit />} />
        <Route path="/audit/runs" element={<AuditRuns />} />
        <Route path="/audit/runs/:id" element={<AuditDetail />} />
        {/* Assessment */}
        <Route path="/assess" element={<Assess />} />
        <Route path="/assess/:framework" element={<Assess />} />
        {/* Frameworks */}
        <Route path="/frameworks" element={<Frameworks />} />
        <Route path="/frameworks/:id" element={<FrameworkDetail />} />
        {/* Policy */}
        <Route path="/policy" element={<Policy />} />
        {/* Evaluate */}
        <Route path="/evaluate" element={<Evaluate />} />
        {/* Progress & Portfolio */}
        <Route path="/my-progress" element={<MyProgress />} />
        <Route path="/portfolio" element={<Portfolio />} />
        <Route path="/badges" element={<Badges />} />
        {/* Auth */}
        <Route path="/auth" element={<Auth />} />
        {/* Catch-all */}
        <Route path="*" element={<NotFound />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;
