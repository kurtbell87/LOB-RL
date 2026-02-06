#!/usr/bin/env bash
# .claude/hooks/pre-tool-use.sh
#
# Blocks modification of test files during GREEN phase.
# The TDD_PHASE env var is set by the orchestrator script.
#
# Hook receives tool name and input via environment variables:
#   CLAUDE_TOOL_NAME  -- the tool being invoked (Edit, Write, Bash, etc.)
#   CLAUDE_TOOL_INPUT -- JSON string of the tool's input parameters

set -euo pipefail

# Only enforce during green phase.
if [[ "${TDD_PHASE:-}" != "green" ]]; then
  exit 0
fi

# Patterns that identify test files (C++ and Python)
TEST_PATTERNS='(test_[^/]*\.|[^/]*_test\.|[^/]*\.test\.|[^/]*\.spec\.|/tests/|/test/|/__tests__/|/spec/|/python/tests/)'

# --- Block 1: Permission escalation commands ---
if [[ "$CLAUDE_TOOL_NAME" == "Bash" ]]; then
  INPUT="$CLAUDE_TOOL_INPUT"

  # Block permission/ownership changes
  if echo "$INPUT" | grep -qEi '(chmod|chown|sudo|doas|install\s)'; then
    echo "BLOCKED: Permission-modifying commands are not allowed during GREEN phase." >&2
    echo "   Test files are read-only by design. Implement to satisfy them." >&2
    exit 1
  fi

  # Block git commands that could revert test files
  if echo "$INPUT" | grep -qEi 'git\s+(checkout|restore|stash|reset)\s'; then
    echo "BLOCKED: Git revert commands are not allowed during GREEN phase." >&2
    echo "   Test files must not be reverted or modified." >&2
    exit 1
  fi

  # Block direct writes to test files via bash
  if echo "$INPUT" | grep -qE "$TEST_PATTERNS"; then
    if echo "$INPUT" | grep -qEi '(>|tee|sed\s+-i|awk.*-i|perl\s+-[pi]|mv\s|cp\s.*>|rm\s)'; then
      echo "BLOCKED: Cannot modify test files via shell commands during GREEN phase." >&2
      exit 1
    fi
  fi
fi

# --- Block 2: Direct file writes to test files ---
if [[ "$CLAUDE_TOOL_NAME" == "Edit" || "$CLAUDE_TOOL_NAME" == "Write" || "$CLAUDE_TOOL_NAME" == "MultiEdit" ]]; then
  if echo "$CLAUDE_TOOL_INPUT" | grep -qE "$TEST_PATTERNS"; then
    echo "BLOCKED: Cannot edit test files during GREEN phase." >&2
    echo "   Tests are your specification. Implement code to satisfy them." >&2
    exit 1
  fi
fi

exit 0
