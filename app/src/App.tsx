import { lazy, Suspense } from "react";
import { BrowserRouter, Routes, Route } from "react-router-dom";
import ErrorBoundary from "./components/ErrorBoundary";
import Copilot from "./components/Copilot";

const Hub = lazy(() => import("./pages/Hub"));
const Audit = lazy(() => import("./pages/Audit"));
const AuditRuns = lazy(() => import("./pages/AuditRuns"));
const AuditDetail = lazy(() => import("./pages/AuditDetail"));
const Evaluate = lazy(() => import("./pages/Evaluate"));
const Frameworks = lazy(() => import("./pages/Frameworks"));
const FrameworkDetail = lazy(() => import("./pages/FrameworkDetail"));
const Assess = lazy(() => import("./pages/Assess"));
const ScenarioAssess = lazy(() => import("./pages/ScenarioAssess"));
const LearningPath = lazy(() => import("./pages/LearningPath"));
const Policy = lazy(() => import("./pages/Policy"));
const MyProgress = lazy(() => import("./pages/MyProgress"));
const Portfolio = lazy(() => import("./pages/Portfolio"));
const Badges = lazy(() => import("./pages/Badges"));
const Auth = lazy(() => import("./pages/Auth"));
const NotFound = lazy(() => import("./pages/NotFound"));

function App() {
  return (
    <BrowserRouter>
      <ErrorBoundary>
      <Suspense fallback={<div className="min-h-screen bg-background" />}>
        <Routes>
          <Route path="/" element={<Hub />} />
          {/* Audit */}
          <Route path="/audit" element={<Audit />} />
          <Route path="/audit/runs" element={<AuditRuns />} />
          <Route path="/audit/runs/:id" element={<AuditDetail />} />
          {/* Assessment */}
          <Route path="/assess" element={<Assess />} />
          <Route path="/assess/:framework" element={<Assess />} />
          <Route path="/assess/scenario/:framework" element={<ScenarioAssess />} />
          <Route path="/learning-path/:frameworkId" element={<LearningPath />} />
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
      </Suspense>
      <Copilot />
      </ErrorBoundary>
    </BrowserRouter>
  );
}

export default App;
