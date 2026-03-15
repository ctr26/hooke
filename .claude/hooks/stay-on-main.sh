#!/bin/bash
# Block git checkout/switch branch commands — enforce worktree workflow.
# Allows: git checkout -- <file>, git checkout -b, git checkout -B, git checkout --orphan
# Only checks the actual command portion, not heredoc/string content.

set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Extract just the first line / actual command, stripping heredoc bodies and quoted strings.
# Take only text before any heredoc delimiter (<<) or long quoted string.
FIRST_CMD=$(echo "$COMMAND" | head -1)

# Block "git switch" entirely (only used for branch switching)
if echo "$FIRST_CMD" | grep -qE '^\s*git\s+switch\b'; then
  echo "BLOCKED: Stay on main. Use \`git worktree add\` or the \`using-git-worktrees\` skill instead." >&2
  exit 2
fi

# Check for "git checkout" as the actual command (not inside strings)
if echo "$FIRST_CMD" | grep -qE '^\s*git\s+checkout\b'; then
  # Allow file restores: git checkout -- <file>
  if echo "$FIRST_CMD" | grep -qE '^\s*git\s+checkout\s+--\s'; then
    exit 0
  fi
  # Allow branch creation flags: -b, -B, --orphan
  if echo "$FIRST_CMD" | grep -qE '^\s*git\s+checkout\s+(-b|-B|--orphan)\b'; then
    exit 0
  fi
  # Everything else is a branch switch — block it
  echo "BLOCKED: Stay on main. Use \`git worktree add\` or the \`using-git-worktrees\` skill instead." >&2
  exit 2
fi

# Also check chained commands (&&, ||, ;) but not content after heredoc/quotes
# Split on && and ; and check each segment
echo "$FIRST_CMD" | tr ';&' '\n' | while read -r segment; do
  segment=$(echo "$segment" | sed 's/^[|&]*//' | xargs)
  if echo "$segment" | grep -qE '^git\s+switch\b'; then
    echo "BLOCKED: Stay on main. Use \`git worktree add\` or the \`using-git-worktrees\` skill instead." >&2
    exit 2
  fi
  if echo "$segment" | grep -qE '^git\s+checkout\b'; then
    if ! echo "$segment" | grep -qE '^git\s+checkout\s+(--\s|-b\b|-B\b|--orphan\b)'; then
      echo "BLOCKED: Stay on main. Use \`git worktree add\` or the \`using-git-worktrees\` skill instead." >&2
      exit 2
    fi
  fi
done

exit 0
