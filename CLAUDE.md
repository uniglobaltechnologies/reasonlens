# CLAUDE.md — ReasonLens

## System Reference

Platform for AI literacy assessment, safety auditing, and policy generation in higher education.
Merged from LearnAI Scope + GlassRoom Lab, rebuilt on Azure (March 2026).

## Tech Stack

- Frontend: React 19, TypeScript, Vite, Tailwind CSS, Three.js (globe)
- API: 18 Azure Functions (Node.js 20, TypeScript)
- Database: PostgreSQL 16 on Azure Flexible Server (25 tables)
- AI: Azure OpenAI (gpt-5.2) for copilot, policy gen, learning paths
- Audit: PETRI Container App (FastAPI, red-teaming engine)
- Auth: bcrypt + JWT (7-day tokens)
- Infra: Azure Static Web App (frontend) + Azure Functions (API)

## Key Commands

```bash
cd app && npm run dev        # Vite dev server (:5173)
cd api && npm start          # Azure Functions local
npm run build                # Production frontend build
```

## Project Layout

| Path | Purpose |
|------|---------|
| `app/` | React frontend (pages, components, data) |
| `api/` | Azure Functions (18 endpoints) |
| `db/` | PostgreSQL migrations (001-005) |
| `docs/` | Grant applications, handover docs |

## Rules — Read These Every Time

### Verification

1. **NEVER claim a bug is fixed without visual verification.** Use Firefox MCP: navigate, screenshot, confirm. No exceptions.
2. **After ANY UI change**: `navigate_page` → `screenshot_page` → `list_console_messages` → confirm correct.
3. **Before claiming "done"**: run `npm run build`, verify in Firefox MCP. Both.
4. If Firefox MCP is unavailable, state that explicitly — do NOT skip verification silently.

### Debugging Protocol

5. **When a bug is reported, do NOT immediately write fixes.** Follow this sequence:
   1. REPRODUCE: Firefox MCP → navigate → screenshot → console errors.
   2. GATHER EVIDENCE: Read logs, check `git diff`, grep for the component.
   3. HYPOTHESIZE: List 3 ranked hypotheses with evidence for/against.
   4. VERIFY TOP HYPOTHESIS: Minimal diagnostic test, screenshot result.
   5. Only after confirming root cause, propose a fix.
6. Never shotgun-apply multiple speculative fixes.

### Deployment

7. Frontend deploys via Azure SWA (push to `main` triggers CI/CD).
8. API deploys via Azure Functions CI/CD.
9. **After deploy**: navigate to production URL in Firefox MCP, screenshot, confirm live.

### Infrastructure

10. **Azure DB has IP firewall rules.** If direct connection fails, use SSH tunnel. Don't retry direct connections.
11. **Framework data lives in TWO places** — `app/src/data/frameworks.ts` (UI) and `api/src/shared/framework-context.ts` (API). Must keep in sync manually.

### Code Quality

12. Do not add features, refactor, or "improve" beyond what was asked.
13. All 22 frameworks must stay in sync between frontend and API.
14. PETRI callbacks use HMAC + timestamp with 5-minute drift tolerance.

### Design System

15. AIFGE brand colors are defined as CSS vars in `index.css` (`--aifge-navy`, `--aifge-teal`, `--aifge-orange`, `--aifge-plum`) and registered in `tailwind.config.ts`. Use Tailwind tokens (`bg-aifge-navy`, `text-aifge-teal`, etc.) — never hardcode hex values.
16. Shared nav/AIFGE link arrays live in `app/src/lib/constants.ts`. Import from there — do not duplicate.
17. CTA gradient is `bg-gradient-cta` in Tailwind — do not inline the gradient style.

### Access Control

18. The site is publicly accessible (no password gate). Only THE DMI routes (`/the-dmi`, `/the-dmi/interpretation/:id`, `/assess/scenario/maturity-the`) are gated behind `PasswordGate` (password: "earlyaccess").

### Autonomy

19. Do not ask questions answerable from the codebase. Read first.
20. When the user references a specific item, confirm which one if ambiguous.

## Infrastructure Quick Reference

| Service | URL / Host |
|---------|-----------|
| Frontend | `reasonlens.com` → Azure SWA |
| API | `reasonlens-api.azurewebsites.net` |
| Database | Azure PostgreSQL Flexible Server (via `DATABASE_URL`) |
| PETRI | Container App (red-teaming audit engine) |
| Parent site | `aiforglobaleducation.org` (WordPress, Hostinger) |

## MCP Servers

- **Firefox DevTools** — browser automation, screenshots, navigation. USE THIS for all verification.
- **Figma Console** — design generation and component management.
