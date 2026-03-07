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
**Live frontend**: `https://purple-hill-0a1de9703.1.azurestaticapps.net`
**Live API**: `https://reasonlens-api.azurewebsites.net/api/`

---

## What Is Complete (Do Not Rebuild)

### Database
- 25 tables on `reasonlens-db.postgres.database.azure.com` (database: `reasonlens`, user: `rladmin`)
- All migrations applied: `db/001` through `db/005`
- Data migrated from original Supabase projects (profiles, runs, reports etc.)
- Schema is at `db/001_unified_schema.sql`

### API (16 Azure Functions)
All deployed to `reasonlens-api` (Consumption plan, Node.js 20):

| Function | What it does |
|---|---|
| `auth` | Signup/login (bcrypt + JWT) + me endpoint |
| `assessments` | Assessment results CRUD |
| `audit-runs` | Run list and detail retrieval |
| `benchmark-callback` | Receives benchmark results via HMAC-protected POST |
| `check-badge-criteria` | Rule-based badge eligibility |
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

### Frontend (14 routes)
React 19, Vite 7, Tailwind v3. Deployed to `reasonlens-app` (Azure Static Web Apps):

- Hub, Audit (Simple + Pro), AuditRuns, AuditDetail, Evaluate, Frameworks, FrameworkDetail, Assess, Policy, MyProgress, Portfolio, Badges, Auth

**Copilot widget** (`app/src/components/Copilot.tsx`) — floating chat, all pages, SSE streaming to `/copilot-chat`, context-aware by route.

**Code splitting** — all pages lazy-loaded. Build output: main bundle ~192KB (down from 1.1MB), framework data chunk ~862KB (unavoidable — it's 22 large data definitions).

### CI/CD
`.github/workflows/deploy.yml` — push to `main` deploys API + frontend. **GitHub Actions secrets are NOT yet set** (see Pending section).

---

## What Is Pending (Start Here)

### 1. GitHub Actions Secrets (unblocks CI/CD)
Go to `https://github.com/AI-For-Global-Education/reasonlens/settings/secrets/actions` and add:

| Secret | Where to get it |
|---|---|
| `AZURE_FUNCTIONS_PUBLISH_PROFILE` | Azure Portal → `reasonlens-api` → Get publish profile |
| `SWA_DEPLOYMENT_TOKEN` | Azure Portal → `reasonlens-app` → Manage deployment token |

Once set, push to `main` will auto-deploy everything.

### 2. Azure Function App Env Vars (unblocks all AI features)
Go to Azure Portal → `reasonlens-api` → Configuration → Application settings and add:

| Variable | Value |
|---|---|
| `JWT_SECRET` | Generate a strong random string (32+ chars) |
| `BYOK_ENC_SECRET` | Generate a strong random string (32+ chars) |
| `PETRI_SERVICE_URL` | `https://petri-service.icyplant-d7ce5d44.uksouth.azurecontainerapps.io` |
| `PETRI_CALLBACK_SECRET` | A shared secret — must also be set on the PETRI service side |
| `TOXICITY_SERVICE_URL` | URL of the toxicity scoring service (check GlassRoom Lab infra) |
| `BENCHMARK_SERVICE_URL` | URL of the benchmark runner service (check GlassRoom Lab infra) |
| `AZURE_OPENAI_ENDPOINT` | `https://aige-petri-resource.cognitiveservices.azure.com` |
| `AZURE_OPENAI_API_KEY` | From Azure Portal → `aige-petri-resource` → Keys |
| `AZURE_OPENAI_DEPLOYMENT` | `gpt-5.2` |
| `DATABASE_URL` | Already in `api/local.settings.json` — copy to Azure |

### 3. PETRI Callback Secret
The PETRI service must send `x-signature` (HMAC-SHA256) and `x-timestamp` headers on callbacks. The secret used must match `PETRI_CALLBACK_SECRET`. Check the PETRI Container App config — this may already be wired up in the old GlassRoom Lab deployment.

### 4. Toxicity + Benchmark Services
These were originally Modal deployments in GlassRoom Lab. Check `uniglobaltechnologies/reasonlens` (old repo) or the GlassRoom Lab codebase for the Modal function URLs. They need to be reachable by the Azure Function App.

### 5. Custom Domain
Azure Portal → `reasonlens-app` → Custom domains → Add. DNS CNAME to `purple-hill-0a1de9703.1.azurestaticapps.net`.

---

## Architecture Quick Reference

```
Browser
  └─ Azure Static Web App (westeurope)
       └─ app/src — React 19 + Vite 7 + Tailwind v3
            └─ lib/api.ts — fetch wrapper, SSE streaming, JWT in localStorage

Azure Functions (uksouth, Consumption)
  └─ api/src/functions/* (16 functions)
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
- 16 Azure Functions (auth, assessments, audit orchestration, AI features)
- 14 React pages + routing
- CI/CD via GitHub Actions
- **Copilot floating widget** — `app/src/components/Copilot.tsx` — SSE streaming to `/copilot-chat`, context-aware, page-specific suggested prompts
- **Code splitting** — lazy page loading via `React.lazy`, `manualChunks` in `vite.config.ts`, main bundle 192KB (was 1.1MB)
- Updated `README.md` and this `HANDOVER.md`
