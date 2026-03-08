# ReasonLens — Agent Handover

> Read this before touching anything. It covers what exists, what works, what doesn't, and what to do next.

---

## What This Is

ReasonLens is a unified platform built by merging two existing products:

- **LearnAI Scope** (`/Users/catorolea/Documents/GitHub/learn-ai-scope/`) — AI literacy framework assessment, portfolios, badges, policy generation. Was on Lovable/Supabase/Gemini.
- **GlassRoom Lab** (`/Users/catorolea/Documents/GitHub/glassroom-lab/`) — AI safety audits via PETRI (red-teaming). Was on Modal/Lovable.

Everything has been rebuilt on Azure (PostgreSQL, Functions, Static Web Apps, OpenAI) with no Supabase, no Modal, no Lovable, no Google AI.

**Local working directory**: `/Users/catorolea/Documents/GitHub/reason-lens/`
**GitHub repo**: `https://github.com/AI-For-Global-Education/reasonlens` (private)
**Live frontend**: `https://reasonlens.com` (also `https://www.reasonlens.com`)
**Live API**: `https://reasonlens-api.azurewebsites.net/api/`

---

## What Is Complete (Do Not Rebuild)

### Database
- 25 tables on `reasonlens-db.postgres.database.azure.com` (database: `reasonlens`, user: `rladmin`)
- All migrations applied: `db/001` through `db/005`
- Data migrated from original Supabase projects (profiles, runs, reports etc.)
- Schema is at `db/001_unified_schema.sql`

### API (18 Azure Functions)
All deployed to `reasonlens-api` (Consumption plan, Node.js 20):

| Function | What it does |
|---|---|
| `auth` | Signup/login (bcrypt + JWT) + me endpoint |
| `assessments` | Assessment results CRUD |
| `audit-runs` | Run list and detail retrieval |
| `benchmark-callback` | Receives benchmark results via HMAC-protected POST |
| `check-badge-criteria` | Rule-based badge eligibility (GET returns all badges with earned status, POST checks criteria) |
| `copilot-chat` | Streaming AI copilot (Azure OpenAI gpt-5.2), personalised from 7 DB queries |
| `framework-recommender` | Tool-calling LLM recommendation from quiz |
| `learning-path-ai` | AI-generated learning paths |
| `parse-audit-intent` | NLP extraction of audit config from natural language |
| `petri-audit-callback` | Receives PETRI results, processes transcripts + posthoc toxicity + benchmarks |
| `policy-generator` | Streaming policy draft (framework-grounded + regulatory) |
| `policy-recommender` | Rule-based gap-to-policy-type mapping |
| `run-benchmark` | Launches CrowS-Pairs / TruthfulQA via BENCHMARK_SERVICE_URL |
| `run-petri-audit` | Launches PETRI audit fire-and-forget, creates audit_runs record |
| `task-evaluator` | AI feasibility scoring (1–5, augment/automate/avoid) |
| `user-api-keys` | BYOK key management (AES-GCM encrypted at rest) |
| `user-progress` | Aggregated progress stats |
| `portfolio` | Portfolio evidence items CRUD (GET/POST/DELETE) |

### Frontend (15 routes)
React 19, Vite 7, Tailwind v3. Deployed to `reasonlens-app` (Azure Static Web Apps):

- Hub, Audit (Simple + Pro), AuditRuns, AuditDetail, Evaluate, Frameworks, FrameworkDetail, Assess, LearningPath, Policy, MyProgress, Portfolio, Badges, Auth
- Policy page supports copy, plain text download, and Word (`.docx`) download of generated drafts.

**Copilot widget** (`app/src/components/Copilot.tsx`) — floating chat, all pages, SSE streaming to `/copilot-chat`, context-aware by route.

**Code splitting** — all pages lazy-loaded. Build output: main bundle ~192KB (down from 1.1MB), framework data chunk ~862KB (unavoidable — it's 22 large data definitions).

### CI/CD
`.github/workflows/deploy.yml` — push to `main` deploys API + frontend.

---

## Production Config Status (Verified 2026-03-08)

### Completed
- Azure Function App settings are configured, including AI + PETRI settings.
- `PETRI_SERVICE_URL` now points to the live POST endpoint: `https://petri-service.icyplant-d7ce5d44.uksouth.azurecontainerapps.io/api` (root URL returns 404).
- `TOXICITY_SERVICE_URL` and `BENCHMARK_SERVICE_URL` are configured to:
  - `https://petri-service.icyplant-d7ce5d44.uksouth.azurecontainerapps.io/toxicity`
  - `https://petri-service.icyplant-d7ce5d44.uksouth.azurecontainerapps.io/benchmark`
- PETRI callback secret wiring is confirmed end-to-end:
  - Function App `PETRI_CALLBACK_SECRET` matches Container App `callback-secret`.
- `JWT_SECRET` rotated from dev placeholder to a strong random value (existing sessions invalidated and must re-login).
- GitHub Actions deploy secrets are configured (`AZURE_FUNCTIONS_PUBLISH_PROFILE`, `FUNCTIONS_STORAGE_ACCOUNT_KEY`, `SWA_DEPLOYMENT_TOKEN`).
- Custom domains are configured and `Ready` on `reasonlens-app`:
  - `reasonlens.com`
  - `www.reasonlens.com`
- Function App platform CORS allow-list includes production domains and localhost:
  - `https://reasonlens.com`, `https://www.reasonlens.com`, `https://purple-hill-0a1de9703.1.azurestaticapps.net`, `http://localhost:5173`

## What Is Pending (Start Here)

### 1. Runtime Upgrade (Time-sensitive)
Azure warns Node.js 20 support for Functions reaches end-of-life on **2026-04-30**.
- Plan and perform runtime upgrade to Node.js 24 for `reasonlens-api`.

---

## Architecture Quick Reference

```
Browser
  └─ Azure Static Web App (westeurope)
       └─ app/src — React 19 + Vite 7 + Tailwind v3
            └─ lib/api.ts — fetch wrapper, SSE streaming, JWT in localStorage

Azure Functions (uksouth, Consumption)
  └─ api/src/functions/* (17 functions)
       ├─ shared/db.ts — pg Pool → reasonlens-db.postgres.database.azure.com
       ├─ shared/auth.ts — bcrypt + JWT validation
       ├─ shared/ai.ts — Azure OpenAI streaming (gpt-5.2)
       └─ middleware/hmac.ts — HMAC validation for callbacks

External Services
  ├─ PETRI Container App — icyplant-d7ce5d44.uksouth.azurecontainerapps.io
  │    Receives: scenario configs + model IDs
  │    Sends back: transcripts + judge scores via /petri-audit-callback
  ├─ Toxicity Service — URL in TOXICITY_SERVICE_URL env var
  │    Called by petri-audit-callback for posthoc toxicity scoring
  └─ Benchmark Service — URL in BENCHMARK_SERVICE_URL env var
       Called by petri-audit-callback for CrowS-Pairs / TruthfulQA
```

---

## Key Files to Know

| File | Why it matters |
|---|---|
| `app/src/lib/api.ts` | All HTTP calls from frontend. `apiStream()` for SSE. Token stored as `reasonlens_token` in localStorage. |
| `app/src/App.tsx` | Router + lazy imports + Copilot widget mount point |
| `app/src/components/Copilot.tsx` | Floating AI chat widget — new, added 2026-03-07 |
| `api/src/shared/auth.ts` | `validateToken`, `requireAuth`, `requireRole` — used by every protected function |
| `api/src/shared/framework-context.ts` | LLM-optimised text of all 22 frameworks — injected into AI prompts |
| `api/src/shared/prompt-preamble.ts` | Shared AI identity, tone, framework name enum |
| `api/src/functions/petri-audit-callback.ts` | Most complex function — parses PETRI results, runs posthoc toxicity + benchmarks |
| `db/001_unified_schema.sql` | Full DB schema — read this to understand data model |

---

## Gotchas

1. **Framework data is in two places**. `app/src/data/frameworks.ts` (frontend, with UI fields) and `api/src/shared/framework-context.ts` (API, LLM-optimised text). If you update frameworks, update both.

2. **Auth is bcrypt + JWT — NOT Supabase, NOT Azure AD B2C**. Token is stored in `localStorage` as `reasonlens_token`. 7-day expiry. There is no refresh token.

3. **PETRI callbacks use HMAC + timestamp**. The callback handler rejects requests with timestamps older than 5 minutes (`MAX_TIMESTAMP_DRIFT_SECONDS = 300` in `hmac.ts`). If PETRI is slow this can cause missed callbacks.

4. **The `runner` role is required to call `/run-petri-audit`**. Users need `runner` or `admin` in `user_roles`. Seed this manually or build an admin UI.

5. **Framework data chunk is 862KB** in the build. This is unavoidable without splitting the framework data files themselves. The main app bundle is 192KB — that's fine.

6. **`local.settings.json` has the DB connection string in plaintext**. Do not commit changes to this file with real secrets.

7. **The old repos** (`uniglobaltechnologies/reasonlens` at `/Users/catorolea/Documents/GitHub/reasonlens/`) is the OLD GlassRoom Lab — not this project. Do not confuse them.

8. **CORS is strict allow-list based**. Ensure `ALLOWED_ORIGINS` in Function App settings includes all frontend domains (at least `https://purple-hill-0a1de9703.1.azurestaticapps.net`, `https://reasonlens.com`, and `https://www.reasonlens.com`).

---

## Source Projects (for reference only — do not modify)

| Project | Local path | Original stack |
|---|---|---|
| LearnAI Scope | `/Users/catorolea/Documents/GitHub/learn-ai-scope/` | Lovable + Supabase + Gemini |
| GlassRoom Lab | `/Users/catorolea/Documents/GitHub/glassroom-lab/` | Lovable + Modal + Supabase |

---

## What Was Done in the Last Session (2026-03-07)

- Built entire platform from scratch (merged two products)
- 25-table PostgreSQL schema + data migration from Supabase
- 17 Azure Functions (auth, assessments, audit orchestration, AI features)
- 14 React pages + routing
- CI/CD via GitHub Actions
- **Copilot floating widget** — `app/src/components/Copilot.tsx` — SSE streaming to `/copilot-chat`, context-aware, page-specific suggested prompts
- **Code splitting** — lazy page loading via `React.lazy`, `manualChunks` in `vite.config.ts`, main bundle 192KB (was 1.1MB)
- Updated `README.md` and this `HANDOVER.md`

## Recent Updates (2026-03-08)

- Added custom domain support end-to-end:
  - Static Web App hostnames live (`reasonlens.com`, `www.reasonlens.com`)
  - Function App CORS allow-list includes production domains
- Fixed CORS handling in API responses (`corsHeaders(req)` across functions) to reflect request origin correctly.
- Assessment flow now persists results to `/assessments` before showing results.
- Added `/learning-path/:frameworkId` route and page to generate/view recommendations from `/learning-path-ai`.
- `My Progress` now reads live stats from `/user-progress` instead of placeholder zeros.
- Simple Audit flow now launches by `scenario_pack` (backend supports `scenario_ids` or `scenario_pack`).
- Policy page now supports Word (`.docx`) export in addition to copy and plain text download.

## QA Sweep (2026-03-08)

Code review identified and fixed 16 issues across 12 files:

### Critical (broken features fixed)
1. **AuditRuns page** (`app/src/pages/AuditRuns.tsx`) — was never fetching data (TODO stub). Now calls `GET /audit-runs` with auth check.
2. **Badges page** (`app/src/pages/Badges.tsx`) — was fully hardcoded with 6 badges all `earned: false`. Now fetches live status from `GET /check-badge-criteria`.
3. **Portfolio page** (`app/src/pages/Portfolio.tsx`) — "Add Evidence" button did nothing. Built full add-evidence form + item list + delete. New API endpoint: `api/src/functions/portfolio.ts` (GET/POST/DELETE).
4. **MyProgress empty state** (`app/src/pages/MyProgress.tsx`) — "Complete an assessment" CTA always showed, even after completing assessments. Now conditional.

### High (data loss / incorrect behaviour)
5. **Policy text download memory leak** (`app/src/pages/Policy.tsx`) — blob URL never revoked. Added `URL.revokeObjectURL()`.
6. **Word export incomplete** (`app/src/pages/Policy.tsx`) — only handled `#`/`##` headings. Added `###`, bullet lists, numbered lists, bold, italic support.
7. **SSE streaming drops last line** (`app/src/lib/api.ts`) — remaining buffer never processed after stream ends. Added buffer flush.
8. **Expired JWT tokens** (`app/src/lib/api.ts`) — `isAuthenticated()` only checked localStorage existence. Now decodes JWT `exp` claim; clears token on 401 responses.

### Medium (UX)
9. **AuditDetail no polling** (`app/src/pages/AuditDetail.tsx`) — page fetched once and never refreshed. Added 8-second polling while status is `running`/`queued`.
10. **LearningPath blank when no data** (`app/src/pages/LearningPath.tsx`) — no empty state. Added fallback with link to start assessment.
11. **SimpleAuditChat suggested prompts** (`app/src/components/audit/SimpleAuditChat.tsx`) — clicking only set input text, didn't submit. Now auto-sends.
12. **Copilot conversation reset** (`app/src/components/Copilot.tsx`) — reset on every URL change. Now only resets on top-level section change.

### Low (defensive)
13. **CORS origin leak** (`api/src/middleware/cors.ts`) — returned first allowed origin for unrecognised requests. Now returns empty string.
14. **BYOK lookup used raw model IDs** (`api/src/functions/run-petri-audit.ts`) — DB query used un-normalised IDs. Now uses normalised IDs.
15. **PETRI failure leaves run stuck** (`api/src/functions/run-petri-audit.ts`) — fire-and-forget only logged errors. Now marks run as `failed` in DB.
16. **Message ID collisions** (`app/src/components/audit/SimpleAuditChat.tsx`) — used `Date.now()`. Replaced with `crypto.randomUUID()`.
