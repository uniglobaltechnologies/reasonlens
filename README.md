# ReasonLens

Unified AI literacy, safety evaluation, and policy platform.

Merges LearnAI Scope (framework assessment, portfolios, policy generation) with GlassRoom Lab (AI safety audits via PETRI) into a single product.

## Structure

```
api/          Azure Functions (Node.js/TypeScript) — backend API
app/          React frontend (TBD)
db/           Database migrations and seed data
infra/        Infrastructure as Code (TBD)
```

## Azure Resources (rg-reasonlens)

- **PostgreSQL**: `reasonlens-db.postgres.database.azure.com` (database: `reasonlens`)
- **Functions**: `reasonlens-api` (Consumption plan, Node.js 20)
- **Container Apps**: PETRI service (Modal compute backend)

## API Endpoints

Base URL: `https://reasonlens-api.azurewebsites.net/api/`

### Framework & Assessment (from LearnAI Scope)
| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/framework-recommender` | POST | None | AI-powered framework recommendation quiz |
| `/task-evaluator` | POST | None | "Can AI do this?" evaluation |
| `/copilot-chat` | POST | Optional JWT | Streaming AI copilot with user context |
| `/policy-generator` | POST | JWT | Streaming AI policy draft generation |
| `/policy-recommender` | POST | JWT | Rule-based policy gap recommendations |
| `/check-badge-criteria` | POST | JWT | Badge eligibility checking |
| `/learning-path-ai` | POST | JWT | AI-powered personalised learning paths |

### AI Safety Audits (from GlassRoom Lab)
| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/run-petri-audit` | POST | JWT + role | Launch PETRI safety audit via Modal |
| `/petri-audit-callback` | POST | HMAC | Receive audit results from Modal |
| `/run-benchmark` | POST | JWT + admin | Launch CrowS-Pairs/TruthfulQA benchmarks |
| `/benchmark-callback` | POST | HMAC | Receive benchmark results from Modal |
| `/parse-audit-intent` | POST | JWT | Extract audit intent from natural language |

### Configuration
| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/user-api-keys` | GET/POST/DELETE | JWT | BYOK API key management (AES-GCM encrypted) |

## Development

```bash
cd api
npm install
npm run build
npm start        # Requires Azure Functions Core Tools v4
```

## Deployment

```bash
cd api
npm run build
func azure functionapp publish reasonlens-api
```
