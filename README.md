# ReasonLens

Unified AI literacy, safety evaluation, and policy platform for education.

Merges LearnAI Scope (framework assessment, portfolios, policy generation) with GlassRoom Lab (AI safety audits via PETRI) into a single product.

**Live**: [https://purple-hill-0a1de9703.1.azurestaticapps.net](https://purple-hill-0a1de9703.1.azurestaticapps.net)

## Structure

```
api/          Azure Functions (Node.js/TypeScript) — 17 endpoints
app/          React + Vite + Tailwind frontend — 16 routes
db/           Database migrations (5) and seed data
.github/      CI/CD workflows
```

## Azure Resources (rg-reasonlens)

| Resource | Type | Location |
|---|---|---|
| `reasonlens-db` | PostgreSQL Flexible Server (B1ms, PG16) | uksouth |
| `reasonlens-api` | Function App (Consumption, Node.js 20) | uksouth |
| `reasonlens-app` | Static Web App (Free) | westeurope |
| `petri-service` | Container App (PETRI audit engine) | uksouth |
| `aigepetricr` | Container Registry | uksouth |
| `aige-petri-resource` | Azure OpenAI (gpt-5.2) | eastus2 |

## API Endpoints

Base URL: `https://reasonlens-api.azurewebsites.net/api/`

### Auth
| Endpoint | Method | Description |
|---|---|---|
| `/auth?action=signup` | POST | Create account (email + password) |
| `/auth?action=login` | POST | Sign in → JWT (7-day expiry) |
| `/auth?action=me` | GET | Get current user from JWT |

### AI-Powered Features
| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/framework-recommender` | POST | None | Framework recommendation quiz |
| `/task-evaluator` | POST | None | "Can AI do this?" evaluation |
| `/copilot-chat` | POST | Optional | Streaming AI copilot |
| `/policy-generator` | POST | JWT | Streaming policy draft generation |
| `/learning-path-ai` | POST | JWT | Personalised learning recommendations |

### Data & Logic
| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/assessments` | GET/POST | JWT | Assessment results CRUD |
| `/audit-runs` | GET | JWT | Audit run detail with transcripts |
| `/user-progress` | GET | JWT | Aggregated progress stats |
| `/policy-recommender` | POST | JWT | Rule-based policy gap analysis |
| `/check-badge-criteria` | POST | JWT | Badge eligibility |
| `/user-api-keys` | GET/POST/DELETE | JWT | BYOK key management |

### Audit Orchestration
| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/run-petri-audit` | POST | JWT + role | Launch PETRI safety audit |
| `/petri-audit-callback` | POST | HMAC | Receive audit results |
| `/run-benchmark` | POST | JWT + admin | Launch bias/truthfulness benchmarks |
| `/benchmark-callback` | POST | HMAC | Receive benchmark results |
| `/parse-audit-intent` | POST | JWT | Extract audit intent from natural language |

## Frontend Routes

| Route | Page |
|---|---|
| `/` | Hub — 5 action pathway cards |
| `/audit` | Test an AI Tool (Simple chat + Pro wizard) |
| `/audit/runs` | Audit history |
| `/audit/runs/:id` | Audit detail (scores, transcripts, posthoc) |
| `/evaluate` | Can AI Do This? (task evaluator) |
| `/frameworks` | Explore 22 frameworks |
| `/frameworks/:id` | Framework detail |
| `/assess` | Assess Your AI Readiness (framework picker) |
| `/assess/:framework` | Self-assessment quiz |
| `/policy` | Generate a Policy (3-step wizard) |
| `/my-progress` | Progress overview |
| `/portfolio` | Evidence portfolio |
| `/badges` | Badge collection |
| `/auth` | Sign in / Sign up |

## Development

```bash
# API
cd api && npm install && npm run build && npm start

# Frontend
cd app && npm install && npx vite dev
```

## Deployment

CI/CD via GitHub Actions on push to `main`. Manual deploy:

```bash
# API
cd api && npm run build && func azure functionapp publish reasonlens-api

# Frontend
cd app && npx vite build && cp staticwebapp.config.json dist/
swa deploy dist --deployment-token $SWA_TOKEN --env production
```

## Tech Stack

- **Database**: Azure PostgreSQL 16 (25 tables, pgcrypto)
- **API**: Azure Functions v4, Node.js 20, TypeScript
- **AI**: Azure OpenAI gpt-5.2
- **Frontend**: React 18, Vite, Tailwind CSS v3, React Router
- **Auth**: bcrypt + JWT
- **Audit Engine**: PETRI via Azure Container Apps
- **Hosting**: Azure Static Web Apps
