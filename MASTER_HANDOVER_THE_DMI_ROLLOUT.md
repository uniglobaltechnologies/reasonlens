# THE DMI Production Rollout — Master Handover

> Scope: the March 18, 2026 production rollout of the Times Higher Education Digital Maturity Index scenario assessment refresh.

---

## Executive Summary

The THE Digital Maturity Index rollout is live in production.

- API deployed to `reasonlens-api`
- Frontend deployed to `reasonlens-app`
- Database migrations `db/011`, `db/012`, and refreshed `db/009` applied directly to production
- THE scenario bank now has `120` active scenarios across `20` child dimensions and `3` adjacent maturity boundaries
- End-to-end smoke test passed against production, then the disposable test account was deleted

This rollout was executed directly from the local workspace using Azure CLI, Kudu zip deploy, the SWA CLI via `npx`, and ad hoc Node/`pg` scripts for SQL application and verification.

---

## What Changed

### Product / Methodology

- THE moved from the earlier `40`-item live scenario bank to a `120`-item adjacent-boundary bank
- Active boundaries in production are now:
  - `incidental-intentional`: `40`
  - `intentional-integrated`: `40`
  - `integrated-optimised`: `40`
- THE onboarding now requires `7` institutional context fields before session creation
- THE scenario assessment now presents an estimated duration of `~40 minutes`
- THE scenario scorer is now boundary-aware instead of using the older conservative min-of-all-answers logic

### Database

- Applied [db/011_expand_scenario_context.sql](/Users/catorolea/Documents/GitHub/reason-lens/db/011_expand_scenario_context.sql)
  - Added:
    - `institution_size`
    - `funding_model`
    - `respondent_role`
    - `respondent_institutional_visibility`
    - `digital_infrastructure_baseline`
- Applied [db/012_normalize_the_assessment_history.sql](/Users/catorolea/Documents/GitHub/reason-lens/db/012_normalize_the_assessment_history.sql)
  - Normalized historical THE `assessment_results.framework_name`
  - Normalized historical THE dimension values to canonical `the-*` ids
- Applied refreshed [db/009_seed_the_scenarios.sql](/Users/catorolea/Documents/GitHub/reason-lens/db/009_seed_the_scenarios.sql)
  - Production now contains:
    - `120` active THE scenarios
    - `20` retired legacy THE scenarios
    - `560` THE response rows

### API

Deployed THE-related backend changes including:

- [api/src/functions/scenario-sessions.ts](/Users/catorolea/Documents/GitHub/reason-lens/api/src/functions/scenario-sessions.ts)
  - requires complete THE institutional context
  - context-aware scenario selection
  - returns estimated time
- [api/src/functions/scenario-answers.ts](/Users/catorolea/Documents/GitHub/reason-lens/api/src/functions/scenario-answers.ts)
  - validates scenario membership in the session
  - no longer leaks mapped maturity level in-flight
- [api/src/functions/scenario-session-complete.ts](/Users/catorolea/Documents/GitHub/reason-lens/api/src/functions/scenario-session-complete.ts)
  - writes canonical THE dimension ids
  - writes the real framework name
  - tracks progress at the `20`-dimension level
- [api/src/functions/user-assessment-context.ts](/Users/catorolea/Documents/GitHub/reason-lens/api/src/functions/user-assessment-context.ts)
  - persists the expanded institutional context fields
- [api/src/shared/scenario-scoring.ts](/Users/catorolea/Documents/GitHub/reason-lens/api/src/shared/scenario-scoring.ts)
  - boundary-aware THE scoring
- [api/src/shared/maturity-the.ts](/Users/catorolea/Documents/GitHub/reason-lens/api/src/shared/maturity-the.ts)
  - THE id normalization, boundary normalization, context completeness, and context fit scoring

### Frontend

Deployed THE-related frontend changes including:

- [app/src/components/assessment/ContextOnboarding.tsx](/Users/catorolea/Documents/GitHub/reason-lens/app/src/components/assessment/ContextOnboarding.tsx)
  - institutional THE onboarding fields
- [app/src/pages/ScenarioAssess.tsx](/Users/catorolea/Documents/GitHub/reason-lens/app/src/pages/ScenarioAssess.tsx)
  - context-gated THE session creation
  - duration estimate
- [app/src/pages/Assess.tsx](/Users/catorolea/Documents/GitHub/reason-lens/app/src/pages/Assess.tsx)
  - updated duration copy for scenario assessments

Related build fixes also went live, including:

- [app/tsconfig.app.json](/Users/catorolea/Documents/GitHub/reason-lens/app/tsconfig.app.json)
- [app/src/lib/api.ts](/Users/catorolea/Documents/GitHub/reason-lens/app/src/lib/api.ts)
- [app/src/components/Copilot.tsx](/Users/catorolea/Documents/GitHub/reason-lens/app/src/components/Copilot.tsx)
- [app/src/pages/AuditDetail.tsx](/Users/catorolea/Documents/GitHub/reason-lens/app/src/pages/AuditDetail.tsx)
- [app/src/data/frameworks.ts](/Users/catorolea/Documents/GitHub/reason-lens/app/src/data/frameworks.ts)
- [app/src/data/frameworks-additional.ts](/Users/catorolea/Documents/GitHub/reason-lens/app/src/data/frameworks-additional.ts)

---

## Data Source and Generation

The THE rollout is driven by the checked-in source bundle under:

- [data/the-dmi](/Users/catorolea/Documents/GitHub/reason-lens/data/the-dmi)

The production SQL seed is generated from that bundle by:

- [scripts/generate-the-scenario-seed.mjs](/Users/catorolea/Documents/GitHub/reason-lens/scripts/generate-the-scenario-seed.mjs)

If the bundle changes again, regenerate [db/009_seed_the_scenarios.sql](/Users/catorolea/Documents/GitHub/reason-lens/db/009_seed_the_scenarios.sql) from the script rather than editing the SQL by hand.

---

## Production State After Rollout

### Live Endpoints

- Frontend:
  - `https://reasonlens.com`
  - `https://www.reasonlens.com`
  - `https://purple-hill-0a1de9703.1.azurestaticapps.net`
- API:
  - `https://reasonlens-api.azurewebsites.net/api`
  - Health check passed at `https://reasonlens-api.azurewebsites.net/api/health`

### THE Scenario Bank

Verified in production after rollout:

- `120` active THE scenarios
- `20` retired THE scenarios
- `40` active `incidental-intentional`
- `40` active `intentional-integrated`
- `40` active `integrated-optimised`

### Historical Data Normalization

Post-rollout verification showed:

- `60` THE assessment rows with normalized framework name
- `65` THE assessment rows using canonical `the-*` dimension ids

---

## Deployment Method Used

### API

- Built a clean release copy under `/tmp/reasonlens-api-release`
- Ran `npm ci`
- Ran `npm run build`
- Ran `npm prune --omit=dev`
- Created a zip artifact under `/tmp`
- Deployed to Azure Functions using `az functionapp deployment source config-zip`

### Frontend

- Built a clean release copy under `/tmp/reasonlens-app-release`
- Ran `npm ci`
- Ran `npm run build`
- Copied `staticwebapp.config.json` into `dist/`
- Fetched the live deployment token with:
  - `az staticwebapp secrets list -g rg-reasonlens -n reasonlens-app`
- Deployed using:
  - `npx @azure/static-web-apps-cli deploy dist --deployment-token ... --env production`

### Database

No `psql` binary was available locally, so production SQL was applied through short Node scripts using the existing `pg` driver and the live `DATABASE_URL` pulled from the Function App settings.

---

## Verification Performed

### Build / Test

- API build passed
- Frontend build passed
- New THE scoring test file passed:
  - [api/src/shared/scenario-scoring.test.ts](/Users/catorolea/Documents/GitHub/reason-lens/api/src/shared/scenario-scoring.test.ts)
- One unrelated legacy test still fails:
  - [api/src/shared/auth.test.ts](/Users/catorolea/Documents/GitHub/reason-lens/api/src/shared/auth.test.ts)

### Live Health

- API health endpoint returned healthy
- Frontend root domain returned `HTTP 200`

### End-to-End Smoke Test

A disposable production user was created and then deleted after verification.

Smoke test covered:

- auth signup
- saving all `7` required THE institutional context fields
- starting a THE scenario session
- receiving `120` scenarios
- answering all scenarios
- completing the session
- verifying `20` saved THE dimension results
- verifying canonical `the-*` dimension ids
- verifying `framework_name = THE Digital Maturity Index`
- verifying `framework_progress` persisted as:
  - `progress = 100`
  - `completed_items = 20`
  - `total_items = 20`

Observed smoke-test result:

- session contained `120` scenarios
- completion returned `20` dimension scores
- first scored dimension resolved to canonical `the-tl-strategy`

The disposable account was then deleted from `profiles`, allowing cascades to clean up its assessment/session/context rows.

---

## Important Operational Notes

### 1. Production Is Ahead of GitHub `main`

This is the most important immediate follow-up.

The rollout was deployed from the local working tree, not from a merged commit on `origin/main`.

That means:

- production currently includes code that is not yet represented by GitHub `main`
- a future GitHub Actions deploy from stale `main` could overwrite the live rollout

Immediate recommendation:

1. commit the rollout changes
2. push to `main`
3. confirm GitHub Actions matches the already-live production state

### 2. Node.js 20 Runtime Deadline

Azure Functions warned during deploy that Node.js `20` reaches end-of-support on **April 30, 2026**.

`reasonlens-api` should be upgraded to Node.js `24` on an explicit schedule.

### 3. Licensing / Rights

The local THE bundle guidance includes a rights caution around publishing derivative scenario content.

Before broader public or commercial scale-out, confirm the THE reuse position.

### 4. Rollback Is Not One Command

This rollout replaced the active THE bank with the new `120`-item structure and retired the old legacy-only rows.

If rollback is needed:

- use git history to recover the prior version of [db/009_seed_the_scenarios.sql](/Users/catorolea/Documents/GitHub/reason-lens/db/009_seed_the_scenarios.sql)
- review which overlapping scenario ids were updated in place
- do not assume that simply flipping row statuses is a full rollback

---

## Files Most Relevant To Future Work

- [MASTER_HANDOVER_THE_DMI_ROLLOUT.md](/Users/catorolea/Documents/GitHub/reason-lens/MASTER_HANDOVER_THE_DMI_ROLLOUT.md)
- [HANDOVER.md](/Users/catorolea/Documents/GitHub/reason-lens/HANDOVER.md)
- [data/the-dmi/scenarios.json](/Users/catorolea/Documents/GitHub/reason-lens/data/the-dmi/scenarios.json)
- [scripts/generate-the-scenario-seed.mjs](/Users/catorolea/Documents/GitHub/reason-lens/scripts/generate-the-scenario-seed.mjs)
- [db/009_seed_the_scenarios.sql](/Users/catorolea/Documents/GitHub/reason-lens/db/009_seed_the_scenarios.sql)
- [db/011_expand_scenario_context.sql](/Users/catorolea/Documents/GitHub/reason-lens/db/011_expand_scenario_context.sql)
- [db/012_normalize_the_assessment_history.sql](/Users/catorolea/Documents/GitHub/reason-lens/db/012_normalize_the_assessment_history.sql)
- [api/src/shared/maturity-the.ts](/Users/catorolea/Documents/GitHub/reason-lens/api/src/shared/maturity-the.ts)
- [api/src/shared/scenario-scoring.ts](/Users/catorolea/Documents/GitHub/reason-lens/api/src/shared/scenario-scoring.ts)
- [api/src/functions/scenario-sessions.ts](/Users/catorolea/Documents/GitHub/reason-lens/api/src/functions/scenario-sessions.ts)
- [api/src/functions/scenario-session-complete.ts](/Users/catorolea/Documents/GitHub/reason-lens/api/src/functions/scenario-session-complete.ts)
- [app/src/components/assessment/ContextOnboarding.tsx](/Users/catorolea/Documents/GitHub/reason-lens/app/src/components/assessment/ContextOnboarding.tsx)
- [app/src/pages/ScenarioAssess.tsx](/Users/catorolea/Documents/GitHub/reason-lens/app/src/pages/ScenarioAssess.tsx)

---

## Recommended Next Actions

1. Commit and push the rollout so GitHub matches production.
2. Decide whether to fix or waive [api/src/shared/auth.test.ts](/Users/catorolea/Documents/GitHub/reason-lens/api/src/shared/auth.test.ts).
3. Plan the Function App runtime upgrade to Node.js `24`.
4. Confirm THE licensing/reuse status before wider external publication.
5. If THE content changes again, regenerate [db/009_seed_the_scenarios.sql](/Users/catorolea/Documents/GitHub/reason-lens/db/009_seed_the_scenarios.sql) from [scripts/generate-the-scenario-seed.mjs](/Users/catorolea/Documents/GitHub/reason-lens/scripts/generate-the-scenario-seed.mjs) and repeat verification before reseeding production.
