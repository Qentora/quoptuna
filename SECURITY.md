# Security Policy

## Supported versions

| Version | Supported |
| ------- | --------- |
| 0.1.x   | ✅        |
| < 0.1   | ❌        |

## Reporting a vulnerability

Please **do not open a public issue** for security vulnerabilities.

Instead, report privately via [GitHub Security Advisories](https://github.com/Qentora/quoptuna/security/advisories/new) ("Report a vulnerability"). Include a description, reproduction steps, and the affected version.

You can expect an acknowledgment within a few days. Once a fix is available we will publish an advisory and credit the reporter (unless you prefer to remain anonymous).

## Scope notes

- QuOptuna runs a local web server (`uvx quoptuna`) intended for localhost use; deploying it publicly is at your own risk.
- LLM report generation sends run metadata to the provider you configure (OpenAI/Anthropic/Google) using your own API key; keys are held client-side and forwarded only to generate reports.
