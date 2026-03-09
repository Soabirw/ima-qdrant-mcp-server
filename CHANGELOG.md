# Changelog

All notable changes to ima-qdrant-mcp-server will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-03-09

### Added

- **MCP server** with two tools: `qdrant_store` (embed + upsert) and `qdrant_find` (semantic search)
- **Ollama embeddings** via HTTP API — replaces FastEmbed to avoid GPU crash risk
- **Per-project `.qdrant` file** — YAML config with directory walk-up discovery
- **Collection name sanitization** — rejects path traversal and special characters
- **Configurable search limit cap** — `MAX_SEARCH_LIMIT` env var (default 100)
- **72 pytest tests** — full async coverage with httpx mocks, no network calls
- **MIT license**, .gitignore, scorecard in README
- **Cross-references** to [ima-claude](https://github.com/Soabirw/ima-claude) and `mcp-qdrant` skill

### Version Bump Checklist

When releasing a new version, update:

1. `pyproject.toml` — `version`
2. `CHANGELOG.md` — new entry
