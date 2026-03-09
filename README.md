# qdrant-mcp-server

Custom Qdrant MCP server with Ollama embeddings. Replaces `mcp-server-qdrant` to fix two problems:
- FastEmbed crashes GPU with larger models → uses Ollama instead
- Official server locks to one collection per instance → collection is resolved per-call

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

## Prerequisites

- Qdrant running locally: `docker run -p 6333:6333 qdrant/qdrant`
- Ollama running with an embedding model: `ollama pull nomic-embed-text`
