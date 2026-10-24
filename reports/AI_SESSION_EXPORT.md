# Original AI-session export

The actual native Codex JSONL is preserved byte-for-byte in the local `session-export/` directory with an SHA-256 manifest. The public repository excludes raw sessions. This active-chat snapshot may omit later tool calls and the final response. Run `pwsh -File scripts/refresh_session_export.ps1` after the chat ends to refresh the finalized original. No messages are reconstructed, translated or rewritten. Instructor acceptance of Codex JSONL instead of an OpenCode-specific `session.json` has not been verified.
