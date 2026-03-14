# ima-qdrant-mcp-server

Custom Qdrant MCP server with pluggable embeddings (Ollama or fastembed). Replaces `mcp-server-qdrant` to fix two problems:
- FastEmbed crashes GPU with larger models → Ollama HTTP API as default, fastembed as CPU-only alternative
- Official server locks to one collection per instance → collection is resolved per-call

Built as the semantic memory backend for [ima-claude](https://github.com/Soabirw/ima-claude) — IMA's Claude Code skills plugin. The `mcp-qdrant` skill in ima-claude provides usage patterns, decision logic, and integration guidance for this server.

## Scorecard

| Category | Grade | Notes |
|----------|-------|-------|
| Code Standards | 🟢 B | Clean FP separation; minor impurities in config |
| Security | 🟢 A | Collection name sanitized; configurable search limit cap |
| Test Coverage | 🟢 B | 79 pytest tests, async mocks, no network calls |
| Documentation | 🟢 B | Good setup/config docs; missing API signatures |
| Maintainability | 🟢 A | 230 lines, clean module boundaries, minimal deps |

> Last reviewed: 2026-03-09 · Skills: py-fp

## Tools

- **qdrant_store** — embed text and upsert into a Qdrant collection
- **qdrant_find** — semantic search across a collection

## Configuration

Collection and URLs resolve in this priority order: **tool argument > env vars > `.qdrant` file > defaults**

### `.qdrant` file (project-level)

Copy `.qdrant.example` to `.qdrant` in your project root:

```yaml
collection: my-project-knowledge
# embedding_provider: ollama  # or "fastembed"
# qdrant_url: http://localhost:6333
# ollama_url: http://localhost:11434
# embedding_model: nomic-embed-text
# vector_size: 768
```

The server walks up from cwd to find the file (like `.gitignore` discovery).

### Environment variables

| Variable | Default | Notes |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | |
| `COLLECTION_NAME` | `ima-knowledge` | |
| `EMBEDDING_PROVIDER` | `ollama` | `"ollama"` or `"fastembed"` |
| `OLLAMA_URL` | `http://localhost:11434` | Only used with ollama provider |
| `EMBEDDING_MODEL` | per provider | ollama: `nomic-embed-text`, fastembed: `BAAI/bge-small-en-v1.5` |
| `VECTOR_SIZE` | per provider | ollama: 768, fastembed: 384 |

## Running

```bash
# Install (Ollama provider)
pip install -e .

# Install (fastembed provider — CPU-only, no Ollama needed)
pip install -e ".[fastembed]"

# Run as MCP server (stdio transport)
qdrant-mcp

# Or directly
python -m qdrant_mcp.server
```

### Claude Desktop / MCP config

```json
{
  "mcpServers": {
    "qdrant-memory": {
      "command": "qdrant-mcp",
      "env": {
        "COLLECTION_NAME": "ima-knowledge"
      }
    }
  }
}
```

## Testing

```bash
pip install -e ".[test]"
pytest -v
```

## Prerequisites

- Qdrant running locally: `docker run -p 6333:6333 qdrant/qdrant`
- **Ollama provider**: Ollama running with an embedding model: `ollama pull nomic-embed-text`
- **fastembed provider**: No external services needed — runs CPU-only ONNX models locally

## Related

- [ima-claude](https://github.com/Soabirw/ima-claude) — Claude Code skills plugin that includes the `mcp-qdrant` skill for this server
- [mcp-qdrant skill](https://github.com/Soabirw/ima-claude/tree/main/plugins/ima-claude/skills/mcp-qdrant) — usage patterns, decision logic (Qdrant vs Vestige vs Serena), and integration guidance

## License

MIT
