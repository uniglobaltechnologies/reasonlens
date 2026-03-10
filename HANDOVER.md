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
  └─ api/src/functions/* (18 functions)
       ├─ shared/db.ts — pg Pool → reasonlens-db.postgres.database.azure.com
       ├─ shared/auth.ts — bcrypt + JWT validation
       ├─ shared/ai.ts — Azure OpenAI streaming (gpt-5.2)
       └─ middleware/hmac.ts — HMAC validation for callbacks

External Services
  ├─ PETRI Container App v2 — icyplant-d7ce5d44.uksouth.azurecontainerapps.io
  │    Image: aigepetricr.azurecr.io/petri-service:v2
  │    Wrapper: FastAPI (service/main.py) → inspect eval petri/audit
  │    Receives: scenario configs + model IDs (POST /api → 202)
  │    Sends back: transcripts + judge scores via HMAC-signed callback
  │    Judge model: openai/azure/gpt-5.2 (Inspect format for Azure OpenAI)
  ├─ Toxicity Service — same container, POST /toxicity (placeholder)
  └─ Benchmark Service — same container, POST /benchmark (placeholder)
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

## Session Updates (2026-03-08 to 2026-03-09)

### Hub Landing Page Redesign
- **Commit**: `eb1db54` — Full rewrite of `app/src/pages/Hub.tsx` (~380 lines)
- UNESCO/OECD institutional style: hero with gradient + floating SVG, animated counters (22 Frameworks, 6 Policy Types, 4 Regions, 5 Pathways), enhanced action pathway cards with colored top borders, timeline "How It Works", trust bar with framework source badges (UNESCO, OECD, JISC, ISTE, etc.), CTA section, enhanced footer
- Custom hooks: `useScrollFadeIn()` (IntersectionObserver), `useCountUp(target, duration)` (requestAnimationFrame counter)
- Uses existing CSS from `index.css`: `animate-float`, `fade-in-on-scroll`, `illustration-hover`
- No new dependencies added

### Bug Fixes
- **Learning Path INSERT** (`commit 9649651`): `api/src/functions/learning-path-ai.ts` was missing `recommendations` and `overall_progress` NOT NULL columns in the INSERT statement. Also added UNIQUE constraint on `(user_id, framework_id)` for the ON CONFLICT clause.
- **Login for pre-bcrypt accounts** (`commit 4e41bf8`): Password reset for accounts created before bcrypt migration.

### PETRI v2 Upgrade (2026-03-09)

#### Problem
The PETRI container app was failing all audit runs with two sequential errors:
1. `ValueError: Model API azure of model 'azure/gpt-5.2' not recognized` — Inspect framework expects `openai/azure/<deployment>` format, not `azure/<deployment>`
2. `DEPRECATED: the 'SpanNode' class has been moved to 'inspect_ai.event.EventTreeSpan'` — The container's PETRI v1 code imported from the old `inspect_ai.log` path

#### What Was Done

**1. Model prefix fix** (`api/src/functions/run-petri-audit.ts`):
- Changed judge model from `azure/gpt-5.2` to `openai/azure/gpt-5.2` (line 88-92)
- This is the format Inspect requires for Azure OpenAI models

**2. Azure environment variables** (Container App):
- Added `AZUREAI_OPENAI_BASE_URL` = `https://aige-petri-resource.cognitiveservices.azure.com/`
- Added `AZUREAI_OPENAI_API_KEY` = (secretref: azure-openai-key)
- Added `AZUREAI_OPENAI_API_VERSION` = `2024-12-01-preview`
- These are the env var names that Inspect expects (the old `AZURE_API_BASE`/`AZURE_API_KEY`/`AZURE_API_VERSION` were not recognized)

**3. PETRI v2 container image** (fork: `github.com/cato-rolea/petri`):
- Merged 63 upstream commits from `github.com/safety-research/petri` into the fork (fast-forward)
- Key upstream changes: fixed `inspect_ai.event` imports (was `inspect_ai.log`), new realism filter system, new CLI viewer, updated scorers
- Created `service/main.py` — FastAPI wrapper that accepts `POST /api`, runs `inspect eval petri/audit` as subprocess, sends results via HMAC-signed callback
- Created `Dockerfile` — Python 3.11-slim + PETRI editable install + FastAPI/uvicorn
- Built and pushed to `aigepetricr.azurecr.io/petri-service:v2`
- Updated container app to revision 7 with new image

**4. Score parsing for v2 format** (`api/src/functions/petri-audit-callback.ts`):
- Added `parseScoresFromJson()` function — extracts scores from `metadata.judge_output.scores` (PETRI v2 JSON format)
- Added `parseScores()` wrapper — tries v2 JSON format first, falls back to v1 XML `<scores>` tags
- Updated all call sites from `parseScoresFromXml()` to `parseScores()`

**5. Role-based access fix**:
- The audit endpoint requires `runner` or `admin` role via `requireRole(req, "runner", "admin")`
- Granted both roles to the user account in `user_roles` table

#### Test Result
- Audit ran end-to-end successfully: **181 transcripts** collected
- Status: **Completed** (green checkmark in UI)
- Models: auditor=google/gemini-2.5-flash, target=openai/gpt-4o-mini, judge=openai/azure/gpt-5.2

#### PETRI v2 New Features Available
- **Realism filter**: Detects when a target model realizes it's being evaluated and filters those runs. Parameters: `realism_model`, `realism_filter` (bool), `realism_threshold` (0.0-1.0). The wrapper service supports these but the frontend doesn't expose them yet.
- **CLI viewer**: `petri view --log-dir ./outputs` — Svelte-based transcript viewer (built into the package, not exposed from container)
- **Resources system**: Configurable resources for auditor agents

#### PETRI Architecture Reference
```
Browser → POST /run-petri-audit (Azure Functions)
              │
              ├─ Creates audit_runs record (status: running)
              └─ Fire-and-forget POST to PETRI_SERVICE_URL
                    │
                    ▼
            PETRI Container App (petri-service:v2)
            FastAPI wrapper (service/main.py:8000)
                    │
                    ├─ POST /api → 202 Accepted
                    ├─ Writes seed_instructions.json
                    ├─ Runs: inspect eval petri/audit \
                    │    --model-role auditor=<model> \
                    │    --model-role target=<model> \
                    │    --model-role judge=openai/azure/gpt-5.2 \
                    │    -T seed_instructions=<file> \
                    │    -T max_turns=10 \
                    │    -T transcript_save_dir=<dir>
                    │
                    ├─ Collects transcript JSON files from output dir
                    └─ HMAC-signed POST to callback_url
                          │
                          ▼
                    POST /petri-audit-callback (Azure Functions)
                          │
                          ├─ Validates HMAC signature + timestamp
                          ├─ Updates audit_runs status
                          ├─ Upserts audit_transcripts
                          ├─ Parses judge scores (v2 JSON or v1 XML)
                          ├─ Runs posthoc toxicity (if requested)
                          └─ Launches benchmark runs (if requested)
```

#### Container App Environment Variables
| Variable | Source | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | secretref: openai-key | For target/auditor models |
| `ANTHROPIC_API_KEY` | secretref: anthropic-key | For Claude target/auditor |
| `GOOGLE_API_KEY` | secretref: google-key | For Gemini target/auditor |
| `GEMINI_API_KEY` | secretref: google-key | Alias for Google |
| `PETRI_CALLBACK_SECRET` | secretref: callback-secret | HMAC signing (len=64) |
| `AZUREAI_OPENAI_BASE_URL` | plaintext | Azure OpenAI endpoint for judge |
| `AZUREAI_OPENAI_API_VERSION` | plaintext: 2025-03-01-preview | Azure API version (must be this or later for Responses API) |
| `AZUREAI_OPENAI_API_KEY` | secretref: azure-openai-key | Azure OpenAI key for judge |
| `UVICORN_TIMEOUT_KEEP_ALIVE` | plaintext: 3700 | Long timeout for audit runs |

#### How to Rebuild PETRI Container
```bash
# Clone the fork
git clone https://github.com/cato-rolea/petri /tmp/petri-wrapper

# (Optional) Sync with upstream
cd /tmp/petri-wrapper
git remote add upstream https://github.com/safety-research/petri.git
git fetch upstream main && git merge upstream/main

# Build and push
az acr build --registry aigepetricr --image petri-service:v2 --image petri-service:latest --platform linux/amd64 .

# Update container app
az containerapp update --name petri-service --resource-group rg-reasonlens --image aigepetricr.azurecr.io/petri-service:v2

# Check logs
az containerapp logs show --name petri-service --resource-group rg-reasonlens --type console --tail 30
```

#### PETRI v2 Transcript Format
```json
{
  "metadata": {
    "transcript_id": "...",
    "auditor_model": "google/gemini-2.5-flash",
    "target_model": "openai/gpt-4o-mini",
    "seed_instruction": "...",
    "tags": [],
    "judge_output": {
      "summary": "...",
      "justification": "...",
      "scores": {
        "needs_attention": 1,
        "concerning": 1,
        "eval_awareness": 1,
        "scenario_realism": 1,
        ...37 dimensions total
      },
      "score_descriptions": { ... }
    }
  },
  "events": [ ... ],
  "messages": [ ... ],
  "target_messages": [ ... ]
}
```

### PETRI v2.3.1 Fixes (2026-03-10)

#### What now works
- **Transcripts**: Real multi-turn conversations produced (170KB+), stored in `audit_transcripts.content`
- **Judge scores**: All 38 alignment dimensions scored and stored in `audit_transcripts.judge_scores_json`
- **Responses API**: Fixed by setting `AZUREAI_OPENAI_API_VERSION=2025-03-01-preview` on the container (modern inspect-ai auto-enables Responses API for GPT-5 models; the old `2024-12-01-preview` version rejected it with 400 errors)
- **Parameter name**: Fixed `special_instructions` → `seed_instructions` (PETRI v2 renamed this parameter)
- **Container image**: `aigepetricr.azurecr.io/petri-service:v2.3.1`
- **Case-sensitive posthoc pack matching**: Fixed `.toLowerCase()` in callback handler

#### What was wrong (for learning)
1. `-M responses_api=false` does NOT apply to `--model-role` models — it only applies to the primary `--model`. PETRI uses `--model-role` exclusively, so this flag did nothing.
2. The old parameter name `special_instructions` was silently ignored by inspect-ai (shown as WARNING in logs but easy to miss).
3. The real fix for the Responses API was always just using the right API version (`2025-03-01-preview`), not trying to disable it.

#### What still doesn't work: Posthoc JT toxicity
The posthoc toxicity pack (JT) never triggers despite:
- `posthoc_packs: ["JT"]` stored correctly in `audit_runs`
- Callback returns 200 OK
- Transcripts stored successfully with content

**Root cause investigation needed**: The callback handler at `petri-audit-callback.ts:445-448` should trigger `runPosthocToxicity()` when `incomingStatus === "completed"` and `posthocPacks.length > 0` and `body.transcripts?.length > 0`. All conditions appear met but no `posthoc_pack_results` rows are created.

**Likely causes to check**:
1. The deployed callback code may differ from local — the `.toLowerCase()` fix and v2 score parsing were edited locally but may not have been in the last `func azure functionapp publish` deployment (check this first!)
2. `extractAssistantResponses()` may fail to parse v2 transcripts — the function looks for `messages`, `target_messages`, `conversation.turns` etc. Need to verify the PETRI v2 transcript structure matches one of these paths
3. The `TOXICITY_SERVICE_URL` env var may not be set or the toxicity endpoint may be failing silently
4. App Insights is not collecting traces from Azure Functions — `az monitor app-insights query` returns empty results, making debugging hard

**To debug**:
```bash
# 1. Check if callback code is deployed with fixes
func azure functionapp publish reasonlens-api  # from api/ directory

# 2. Add temporary console.log in petri-audit-callback.ts around line 445:
#    context.log("POSTHOC DEBUG", { posthocPacks, transcriptCount: body.transcripts?.length, incomingStatus });

# 3. Trigger audit and check Function App live logs:
az webapp log tail --name reasonlens-api --resource-group rg-reasonlens

# 4. Check TOXICITY_SERVICE_URL is set:
az functionapp config appsettings list --name reasonlens-api --resource-group rg-reasonlens --query "[?name=='TOXICITY_SERVICE_URL'].value"
```

### Key Environment Changes Made (2026-03-10)
| Setting | Old Value | New Value | Where |
|---|---|---|---|
| `AZUREAI_OPENAI_API_VERSION` | `2024-12-01-preview` | `2025-03-01-preview` | Container App env var |
| Container image | `petri-service:v2` | `petri-service:v2.3.1` | Container App |
| `-T` parameter | `special_instructions=<file>` | `seed_instructions=<file>` | `service/main.py` |
| `-M responses_api=false` | present | **removed** (does nothing for model-roles) | `service/main.py` |

### Verified Working Run
- **Run ID**: `5511c7fc-458c-4662-8c8f-7dfba2c20ef2`
- **Models**: all `openai/azure/gpt-5.2`
- **Transcripts**: 1 (content_len: 171584)
- **Judge scores**: 38 dimensions (admirable:2, concerning:1, needs_attention:2, scenario_realism:7, etc.)
- **Posthoc**: not triggered (see above)

### Pending Items
1. **Fix posthoc JT toxicity** — see investigation notes above
2. **Node.js 24 runtime upgrade** — Azure warns Node.js 20 EOL is 2026-04-30
3. **Realism filter UI** — PETRI v2 realism parameters supported in wrapper but not in frontend
4. **Login broken** — `auth?action=login` returns 500 for `cato.rolea@gmail.com` even after password reset. JWT can be generated locally using `JWT_SECRET` from Azure config as workaround. Password was reset to `Admin123!` via direct DB update but login still fails with Internal Server Error (needs debugging).
5. **Pro audit mode** — Frontend shows "Coming soon" placeholder. Need to build the manual configuration UI.
6. **App Insights not collecting traces** — Function app traces are empty in App Insights queries. May need to configure `APPLICATIONINSIGHTS_CONNECTION_STRING` or check if Application Insights is properly linked.
