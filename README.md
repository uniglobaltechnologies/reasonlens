# ReasonLens

> Unified AI literacy, safety evaluation, and policy platform for education.

Merges **LearnAI Scope** (framework assessment, portfolios, policy generation) with **GlassRoom Lab** (AI safety audits via PETRI) into a single product for higher education institutions.

**GitHub**: `AI-For-Global-Education/reasonlens` (private)
**Live**: `https://reasonlens.com` (also `https://www.reasonlens.com`)
**API**: `https://reasonlens-api.azurewebsites.net/api/`

---

## Repository Structure

```
reason-lens/
├── app/                        React + Vite + Tailwind CSS v3 frontend
│   └── src/
│       ├── App.tsx             Router — lazy-loaded pages + Copilot widget
│       ├── pages/              15 route-level page components
│       ├── components/
│       │   ├── Copilot.tsx     Floating AI chat widget (all pages)
│       │   ├── Header.tsx      Site header + auth state
│       │   ├── ThemeToggle.tsx Dark/light mode
│       │   └── audit/          SimpleAuditChat, ProAuditWizard
│       ├── data/               22 framework definitions + policy/regulatory data
│       └── lib/api.ts          Fetch wrapper, SSE streaming, JWT management
│
├── api/                        Azure Functions v4 (Node.js 20, TypeScript)
│   └── src/
│       ├── functions/          17 Azure Functions
│       ├── shared/
│       │   ├── auth.ts         validateToken, requireAuth, requireRole
│       │   ├── db.ts           PostgreSQL pool (query, queryOne, execute)
│       │   ├── crypto.ts       AES-GCM encryption for BYOK keys
│       │   ├── ai.ts           Azure OpenAI streaming helpers
│       │   ├── framework-context.ts  All 22 frameworks as LLM-optimised text
│       │   └── prompt-preamble.ts    Shared AI identity/tone/framework enum
│       └── middleware/
│           ├── cors.ts         CORS headers
│           └── hmac.ts         HMAC-SHA256 + 5-min replay protection for callbacks
│
├── db/
│   ├── 001_unified_schema.sql  25 tables (9 groups)
│   ├── 002_seed_data.sql       Badges, scenarios, models seed data
│   ├── 003_add_unique_constraints.sql
│   ├── 004_data_migration.py   Migrated data from Supabase (LearnAI Scope)
│   └── 005_auth_columns.sql    password_hash column on profiles
│
└── .github/workflows/
    └── deploy.yml              CI/CD: push to main deploys API + SWA
```

---

## Azure Resources (`rg-reasonlens`)

| Resource | Type | SKU | Location |
|---|---|---|---|
| `reasonlens-db` | PostgreSQL Flexible Server 16 | Burstable B1ms | uksouth |
| `reasonlens-api` | Function App (Node.js 20) | Consumption | uksouth |
| `reasonlens-app` | Static Web App | Free | westeurope |
| `petri-service` | Container App (PETRI audit engine) | — | uksouth |
| `aigepetricr` | Container Registry | — | uksouth |
| `aige-petri-resource` | Azure OpenAI (gpt-5.2) | — | eastus2 |

---

## Database Schema (25 tables)

| Group | Tables |
|---|---|
| 1. Identity & Auth | `profiles`, `user_roles` |
| 2. Framework Assessment | `user_goals`, `assessment_results`, `framework_progress`, `learning_paths` |
| 3. Portfolio & Evidence | `portfolio_items`, `competency_tags`, `portfolio_shares` |
| 4. Policy | `policy_drafts` |
| 5. Chat | `chat_conversations`, `chat_messages` |
| 6. Badges & Achievements | `badges`, `user_badges`, `user_achievements` |
| 7. AI Safety Audits | `audit_runs`, `audit_transcripts`, `scenarios`, `audit_reports`, `posthoc_pack_results`, `benchmark_runs` |
| 8. Configuration | `models`, `user_api_keys`, `audit_log` |
| 9. Bridge | `assessment_evidence` |

**Auth**: bcrypt on `profiles.password_hash` + JWT 7-day tokens signed with `JWT_SECRET`.
**Roles**: `admin`, `educator`, `leader`, `student`, `runner`, `viewer` — stored in `user_roles`, never on `profiles`.
**BYOK**: User API keys AES-GCM encrypted via `BYOK_ENC_SECRET` before storage in `user_api_keys`.

---

## API Endpoints

Base URL: `https://reasonlens-api.azurewebsites.net/api/`

### Auth

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/auth?action=signup` | POST | None | Creates profile, returns JWT |
| `/auth?action=login` | POST | None | Validates bcrypt, returns JWT |
| `/auth?action=me` | GET | JWT | Returns full profile |

### AI-Powered (all stream via SSE)

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/copilot-chat` | POST | Optional | Framework AI copilot. Fetches 7 DB queries to personalise if authenticated. |
| `/policy-generator` | POST | JWT | Streams institution-specific policy draft grounded in frameworks + regulation. |
| `/learning-path-ai` | POST | JWT | AI-generated personalised learning recommendations. |
| `/framework-recommender` | POST | JWT | Tool-calling LLM maps quiz answers to framework recommendations. |
| `/task-evaluator` | POST | JWT | Scores educational tasks for AI feasibility (1-5). Returns augment/automate/avoid. |

### Assessment & Progress

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/assessments` | GET/POST | JWT | Assessment results CRUD |
| `/user-progress` | GET | JWT | Aggregated progress stats |
| `/policy-recommender` | POST | JWT | Rule-based: maps assessment gaps to policy types |
| `/check-badge-criteria` | POST | JWT | Badge eligibility check |

### Audit Orchestration

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/run-petri-audit` | POST | JWT + runner/admin | Launches PETRI safety audit fire-and-forget (supports `scenario_ids` or `scenario_pack`) |
| `/petri-audit-callback` | POST | HMAC | Receives results, processes transcripts + posthoc toxicity + benchmarks |
| `/run-benchmark` | POST | JWT + admin | Launches CrowS-Pairs / TruthfulQA benchmark |
| `/benchmark-callback` | POST | HMAC | Receives benchmark results |
| `/parse-audit-intent` | POST | JWT | NLP extraction of audit config from plain text |
| `/audit-runs` | GET | JWT | Run list + detail (transcripts, posthoc, benchmarks) |

### User Settings

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/user-api-keys` | GET/POST/DELETE | JWT | BYOK API key management (encrypted at rest) |

---

## Frontend Routes

| Route | Page | Notes |
|---|---|---|
| `/` | Hub | 5 action pathway cards |
| `/audit` | Test an AI Tool | Simple chat + Pro wizard modes |
| `/audit/runs` | Audit Run History | — |
| `/audit/runs/:id` | Audit Detail | Scores, transcripts, posthoc, benchmarks |
| `/evaluate` | Can AI Do This? | Task evaluator |
| `/frameworks` | Framework Explorer | All 22 frameworks |
| `/frameworks/:id` | Framework Detail | Dimensions, levels, indicators |
| `/assess` | Framework Picker | — |
| `/assess/:framework` | Self-Assessment Quiz | Per-dimension level selection |
| `/learning-path/:frameworkId` | Learning Path | AI-generated recommendations from saved assessment results |
| `/policy` | Policy Generator | 3-step wizard + copy/text/Word export |
| `/my-progress` | Progress Dashboard | — |
| `/portfolio` | Evidence Portfolio | Upload, tag, share artifacts |
| `/badges` | Badge Collection | — |
| `/auth` | Sign In / Sign Up | — |

**Global**: `Copilot` floating chat widget mounted outside `<Routes>` in `App.tsx`. Visible on all pages except `/auth`. Resets conversation on route change. Sends current page label + framework ID (from URL) as context to `/copilot-chat`.

---

## Framework Data (22 frameworks)

Lives in `app/src/data/`. The API has a parallel LLM-optimised copy in `api/src/shared/framework-context.ts` (no UI fields). **These must be kept in sync manually.**

| File | Content |
|---|---|
| `frameworks.ts` | All 22 framework definitions (~6,000 lines) |
| `frameworks-additional.ts` | Supplementary metadata |
| `framework-types.ts` | TypeScript schema types |
| `digcomp-3-source.json` | DigComp 3.0 official JRC data (CC BY 4.0) |
| `esco-digcomp-mapping.json` | ~65 ESCO skills mapped to DigComp competences |
| `bdc-*.json` (7 files) | JISC Building Digital Capability role profiles |
| `regulatory-context.json` | EU AI Act, UK DfE, US FERPA/NIST provisions |
| `policy-templates.json` | 6 policy type templates |
| `policy-dimension-mapping.json` | Dimension gaps to policy type triggers |

---

## Environment Variables

### API (set in Azure Function App → Configuration)

| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | Yes | PostgreSQL connection string (SSL required) |
| `JWT_SECRET` | Yes | Signs and verifies auth tokens |
| `BYOK_ENC_SECRET` | Yes | AES-GCM key for encrypting user API keys |
| `PETRI_SERVICE_URL` | Yes | PETRI run endpoint (`.../api`) |
| `PETRI_CALLBACK_SECRET` | Yes | HMAC secret for callback validation |
| `TOXICITY_SERVICE_URL` | Yes | Posthoc toxicity endpoint (`.../toxicity`) |
| `BENCHMARK_SERVICE_URL` | Yes | Benchmark endpoint (`.../benchmark`) |
| `AZURE_OPENAI_ENDPOINT` | Yes | Azure OpenAI resource endpoint |
| `AZURE_OPENAI_API_KEY` | Yes | Azure OpenAI API key |
| `AZURE_OPENAI_DEPLOYMENT` | Yes | Deployment name (gpt-5.2) |

### Frontend (build-time)

| Variable | Default | Description |
|---|---|---|
| `VITE_API_URL` | `https://reasonlens-api.azurewebsites.net/api` | API base URL override |

### GitHub Actions Secrets (set in repo Settings → Secrets)

| Secret | Where to get it |
|---|---|
| `AZURE_FUNCTIONS_PUBLISH_PROFILE` | Azure Portal → Function App → Get publish profile |
| `FUNCTIONS_STORAGE_ACCOUNT_KEY` | Azure Portal → Storage Account (`reasonlensfuncstor`) → Access keys |
| `SWA_DEPLOYMENT_TOKEN` | Azure Portal → Static Web App → Manage deployment token |

Helper script (requires `az` + authenticated `gh`):
```bash
./infra/sync-github-actions-secrets.sh
```

---

## Development

```bash
# API
cd api
npm install
# Fill in api/local.settings.json with real values
npm run build
npm start   # http://localhost:7071

# Frontend
cd app
npm install
# Create app/.env.local:  VITE_API_URL=http://localhost:7071/api
npx vite dev  # http://localhost:5173
```

---

## Deployment

Push to `main` triggers CI/CD automatically (requires GitHub Actions secrets to be set).

Manual deploy:
```bash
# API
cd api && npm run build
func azure functionapp publish reasonlens-api

# Frontend
cd app && npx vite build
cp staticwebapp.config.json dist/
swa deploy dist --deployment-token $SWA_DEPLOYMENT_TOKEN --env production
```

---

## Production Checklist

| # | Item | Status |
|---|---|---|
| 1 | All 17 API functions | Done |
| 2 | All 15 frontend routes | Done |
| 3 | 25-table DB schema + data migration | Done |
| 4 | Auth (bcrypt + JWT) | Done |
| 5 | CI/CD pipeline (GitHub Actions) | Done |
| 6 | Copilot floating widget | Done 2026-03-07 |
| 7 | Code splitting + lazy page loading | Done 2026-03-07 |
| 8 | Set Azure Function App env vars | Done 2026-03-08 |
| 9 | Set GitHub Actions secrets | Done 2026-03-08 |
| 10 | Configure PETRI_CALLBACK_SECRET on PETRI service | Done 2026-03-08 (verified match) |
| 11 | Configure TOXICITY_SERVICE_URL + BENCHMARK_SERVICE_URL | Done 2026-03-08 |
| 12 | Custom domain on Static Web App | Done 2026-03-08 (`reasonlens.com`, `www.reasonlens.com`) |
| 13 | Upgrade Function App runtime to Node 24 before 2026-04-30 | Pending |
| 14 | Function App CORS allow-list for custom domains | Done 2026-03-08 |
