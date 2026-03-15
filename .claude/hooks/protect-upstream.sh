#!/bin/bash
# Block agents from pushing code, issues, PRs, or branches to upstream when a fork exists.
# A fork is detected by the presence of an "upstream" remote.

set -euo pipefail

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')
CWD=$(echo "$INPUT" | jq -r '.cwd // empty')

# Only check Bash commands
if [ "$TOOL" != "Bash" ]; then
  exit 0
fi

COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Check if repo has an upstream remote (i.e. is a fork)
HAS_UPSTREAM=false
if [ -n "$CWD" ] && git -C "$CWD" remote 2>/dev/null | grep -q '^upstream$'; then
  HAS_UPSTREAM=true
fi

if [ "$HAS_UPSTREAM" = false ]; then
  exit 0
fi

# Block git push to upstream
if echo "$COMMAND" | grep -qE 'git\s+push\s+upstream'; then
  echo "BLOCKED: Cannot push to upstream remote. This is a fork — push to origin instead." >&2
  exit 2
fi

# Block gh commands targeting upstream
UPSTREAM_URL=$(git -C "$CWD" remote get-url upstream 2>/dev/null || echo "")
UPSTREAM_REPO=""
if [ -n "$UPSTREAM_URL" ]; then
  # Extract owner/repo from URL (handles both HTTPS and SSH)
  UPSTREAM_REPO=$(echo "$UPSTREAM_URL" | sed -E 's|.*[:/]([^/]+/[^/]+?)(\.git)?$|\1|')
fi

# Block issue creation on upstream
if echo "$COMMAND" | grep -qE 'gh\s+issue\s+create'; then
  if [ -n "$UPSTREAM_REPO" ] && echo "$COMMAND" | grep -qF "$UPSTREAM_REPO"; then
    echo "BLOCKED: Cannot create issues on upstream repo ($UPSTREAM_REPO). Create on your fork instead." >&2
    exit 2
  fi
  # Also block if using -R/--repo pointing to upstream
  if echo "$COMMAND" | grep -qE "(-R|--repo)\s+$UPSTREAM_REPO"; then
    echo "BLOCKED: Cannot create issues on upstream repo ($UPSTREAM_REPO)." >&2
    exit 2
  fi
fi

# Block PR creation targeting upstream
if echo "$COMMAND" | grep -qE 'gh\s+pr\s+create'; then
  # Block if explicitly targeting upstream repo
  if [ -n "$UPSTREAM_REPO" ] && echo "$COMMAND" | grep -qF "$UPSTREAM_REPO"; then
    echo "BLOCKED: Cannot create PRs on upstream repo ($UPSTREAM_REPO). Create on your fork instead." >&2
    exit 2
  fi
  if echo "$COMMAND" | grep -qE "(-R|--repo)\s+$UPSTREAM_REPO"; then
    echo "BLOCKED: Cannot create PRs on upstream repo ($UPSTREAM_REPO)." >&2
    exit 2
  fi
fi

exit 0
