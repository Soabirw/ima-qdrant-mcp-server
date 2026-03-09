# ima-qdrant-mcp-server

Custom Qdrant MCP server with Ollama embeddings. Replaces `mcp-server-qdrant` to fix two problems:
- FastEmbed crashes GPU with larger models → uses Ollama HTTP API instead
- Official server locks to one collection per instance → collection is resolved per-call

Built as the semantic memory backend for [ima-claude](https://github.com/Soabirw/ima-claude) — IMA's Claude Code skills plugin. The `mcp-qdrant` skill in ima-claude provides usage patterns, decision logic, and integration guidance for this server.

## Scorecard

| Category | Grade | Notes |
|----------|-------|-------|
| Code Standards | 🟢 B | Clean FP separation; minor impurities in config |
| Security | 🟡 B | No limit cap on search; collection name unsanitized |
| Test Coverage | 🟢 B | 56 pytest tests, async mocks, no network calls |
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
# qdrant_url: http://localhost:6333
# ollama_url: http://localhost:11434
# embedding_model: nomic-embed-text
# vector_size: 768
```

The server walks up from cwd to find the file (like `.gitignore` discovery).

### Environment variables

| Variable | Default |
|---|---|
| `QDRANT_URL` | `http://localhost:6333` |
| `COLLECTION_NAME` | `ima-knowledge` |
| `OLLAMA_URL` | `http://localhost:11434` |
| `EMBEDDING_MODEL` | `nomic-embed-text` |

## Running

```bash
# Install
pip install -e .

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
- Ollama running with an embedding model: `ollama pull nomic-embed-text`

## Related

- [ima-claude](https://github.com/Soabirw/ima-claude) — Claude Code skills plugin that includes the `mcp-qdrant` skill for this server
- [mcp-qdrant skill](https://github.com/Soabirw/ima-claude/tree/main/plugins/ima-claude/skills/mcp-qdrant) — usage patterns, decision logic (Qdrant vs Vestige vs Serena), and integration guidance

## License

MIT
