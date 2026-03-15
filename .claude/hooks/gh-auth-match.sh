#!/bin/bash
# Ensure gh CLI auth matches the repo owner before running gh commands.
# If the active GitHub account doesn't match the repo owner, switch to
# the correct account automatically.

set -euo pipefail

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')

# Only check Bash commands
if [ "$TOOL" != "Bash" ]; then
  exit 0
fi

COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')

# Only check commands that use gh CLI
if ! echo "$COMMAND" | grep -qE '\bgh\s'; then
  exit 0
fi

# Skip gh auth commands (status, switch, login) — they're how you fix mismatches
if echo "$COMMAND" | grep -qE '\bgh\s+auth\b'; then
  exit 0
fi

# Get repo owner from git remote
REMOTE_URL=""
if [ -n "$CWD" ]; then
  REMOTE_URL=$(git -C "$CWD" remote get-url origin 2>/dev/null || echo "")
else
  REMOTE_URL=$(git remote get-url origin 2>/dev/null || echo "")
fi

if [ -z "$REMOTE_URL" ]; then
  exit 0
fi

# Extract owner from remote URL (handles HTTPS and SSH)
REPO_OWNER=$(echo "$REMOTE_URL" | sed -E 's|.*[:/]([^/]+)/[^/]+(\.git)?$|\1|')

if [ -z "$REPO_OWNER" ]; then
  exit 0
fi

# Get current gh auth account
ACTIVE_USER=$(gh api user --jq '.login' 2>/dev/null || echo "")

if [ -z "$ACTIVE_USER" ]; then
  echo "WARNING: Could not determine active GitHub account. Run 'gh auth status' to check." >&2
  exit 0
fi

# Compare (case-insensitive)
if [ "$(echo "$ACTIVE_USER" | tr '[:upper:]' '[:lower:]')" != "$(echo "$REPO_OWNER" | tr '[:upper:]' '[:lower:]')" ]; then
  # Check if the repo owner account is available to switch to
  if gh auth switch --user "$REPO_OWNER" 2>/dev/null; then
    echo "NOTICE: Switched gh auth from '$ACTIVE_USER' to '$REPO_OWNER' to match repo owner." >&2
  else
    echo "BLOCKED: Active GitHub account is '$ACTIVE_USER' but repo belongs to '$REPO_OWNER'. Run 'gh auth login' as '$REPO_OWNER' or 'gh auth switch --user $REPO_OWNER'." >&2
    exit 2
  fi
fi

exit 0
