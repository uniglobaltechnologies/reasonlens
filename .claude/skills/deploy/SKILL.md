# /deploy — Build, Deploy, and Verify

Enforce end-to-end verification before marking any deploy as complete.

## Steps

1. **Pre-deploy checks**
   - Run `cd app && npm run build` — must succeed with zero errors
   - If it fails, fix before proceeding

2. **Deploy**
   - Push to `main` — Azure SWA auto-deploys frontend + API via GitHub Actions
   - State what was deployed

3. **Post-deploy verification (MANDATORY)**
   - Wait 60 seconds for deploy to propagate
   - Use Firefox MCP to navigate to `https://reasonlens.com`
   - Take a screenshot of the affected page(s)
   - Check `list_console_messages` for errors
   - If the change involves interactive elements (assessments, copilot, audit), test at least 2 interactions and screenshot results

4. **Report**
   - Show the screenshots to the user
   - List any console errors found
   - Only mark as complete if screenshots confirm working state
   - If anything is broken, fix and restart from step 1

## Never

- Never say "deployed successfully" without a Firefox screenshot proving it
- Never substitute `curl` for Firefox when verifying UI changes
