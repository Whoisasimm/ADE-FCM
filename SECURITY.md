# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.0.x   | ✅ |
| < 1.0   | ❌ |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability in ADE-FCM:

1. **DO NOT** open a public issue
2. Email the maintainers at research@ade-fcm.org
3. Include a detailed description and steps to reproduce
4. Allow 48 hours for initial response

## Security Measures

This project:
- ❌ Does not hardcode credentials, API keys, or tokens (verified by security audit)
- ❌ Does not contain production secrets (all secrets use environment variables)
- ✅ Uses only container-level paths (no user-specific absolute paths)
- ✅ Encrypts communications via HTTPS for distributed mode
- ✅ Sanitizes all input data before processing

## Report Template

```
Subject: [ADE-FCM Security] Brief description
- Version: 
- Component: 
- Vulnerability Type: 
- Description: 
- Reproduction Steps: 
- Impact: 
- Suggested Fix: 
```
