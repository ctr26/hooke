#!/bin/bash
# Block git checkout/switch branch commands — enforce worktree workflow.
# Allows: git checkout -- <file>, git checkout -b, git checkout -B, git checkout --orphan

set -euo pipefail

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Block "git switch" entirely (only used for branch switching)
if echo "$COMMAND" | grep -qE 'git\s+switch\b'; then
  echo "BLOCKED: Stay on main. Use \`git worktree add\` or the \`using-git-worktrees\` skill instead." >&2
  exit 2
fi

# Check for "git checkout" commands
if echo "$COMMAND" | grep -qE 'git\s+checkout\b'; then
  # Allow file restores: git checkout -- <file>
  if echo "$COMMAND" | grep -qE 'git\s+checkout\s+--\s'; then
    exit 0
  fi
  # Allow branch creation flags: -b, -B, --orphan (still in worktree context but not blocking)
  if echo "$COMMAND" | grep -qE 'git\s+checkout\s+(-b|-B|--orphan)\b'; then
    exit 0
  fi
  # Everything else is a branch switch — block it
  echo "BLOCKED: Stay on main. Use \`git worktree add\` or the \`using-git-worktrees\` skill instead." >&2
  exit 2
fi

exit 0
