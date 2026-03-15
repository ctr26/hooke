#!/bin/bash
# Route styleguide download/chunk commands to the chunk-styleguide skill.
# This hook fires on PreToolUse for Bash commands and checks if the user
# is attempting a manual styleguide operation that should use the skill instead.

set -euo pipefail

INPUT=$(cat)
TOOL=$(echo "$INPUT" | jq -r '.tool_name // empty')

# Only check Bash commands
if [ "$TOOL" != "Bash" ]; then
  exit 0
fi

COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Check for manual styleguide download/chunk attempts
if echo "$COMMAND" | grep -qiE '(styleguide|style.guide|style_guide).*(download|chunk|split|segment)'; then
  echo "NOTICE: Use /chunk-styleguide <url> instead of manual download/chunking. This ensures consistent output structure in .claude/styleguides/." >&2
  exit 0
fi

if echo "$COMMAND" | grep -qiE '(download|curl|wget).*(styleguide|style.guide|pyguide|pep8)'; then
  echo "NOTICE: Use /chunk-styleguide <url> to download and chunk style guides into agent-friendly format." >&2
  exit 0
fi

exit 0
