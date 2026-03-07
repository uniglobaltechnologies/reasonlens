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

## Development

```bash
cd api
npm install
npm run build
npm start        # Requires Azure Functions Core Tools
```
