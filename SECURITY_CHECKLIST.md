# Security Checklist for GitHub Publication

## Completed Fixes

- [x] **`deployment/mlflow/mlflow_config.py`**: Replaced hardcoded `/tmp/` path with `tempfile.gettempdir()`
- [x] **`deployment/docker-compose.yml`**: Replaced hardcoded Jupyter token with `${JUPYTER_TOKEN}` env variable
- [x] **`SECURITY.md`**: Updated to reflect actual security posture (environment-based secrets)

## Pre-Push Verification

- [ ] Verify no hardcoded local paths remain: `grep -rn "D:\\\\" --include="*.py" --include="*.md" --include="*.yml" --include="*.yaml" .`
- [ ] Verify no hardcoded passwords/tokens: `grep -rn "password\|token\|secret\|api_key" --include="*.py" --include="*.yml" --include="*.yaml" . | grep -v "os.getenv\|\.env\|${" | grep -v "tokenize\|Tokenization\|tokenizer"`
- [ ] Verify no email addresses: `grep -rn "[a-zA-Z0-9._%+-]\+@[a-zA-Z0-9.-]\+\.[a-zA-Z]\{2,\}" --include="*.py" --include="*.md" . | grep -v "example\|your@"`
- [ ] Verify `.gitignore` excludes local env files, IDE artifacts, and internal reports
- [ ] Run `git diff --cached` to review staged files for any secrets
- [ ] Run `git secret scan` or `trufflehog` if available

## Before First Push

- [ ] Remove the `origin` remote initially, push to a private test repo first
- [ ] Enable branch protection on `main` once public
- [ ] Add GitHub Secrets for `JUPYTER_TOKEN` and other env vars
- [ ] Enable Dependabot for dependency vulnerability alerts
- [ ] Enable CodeQL scanning
- [ ] Create a `v1.0.0` release with the release notes
- [ ] Upload to Zenodo (link to repo for DOI generation)
- [ ] Add repository topics on GitHub: `fuzzy-clustering`, `unsupervised-learning`, `explainable-ai`, etc.
