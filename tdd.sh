#!/usr/bin/env bash
# tdd.sh -- TDD Workflow Orchestrator for LOB-RL
#
# Usage:
#   ./tdd.sh red   <spec-file>     # Write tests from spec
#   ./tdd.sh green                  # Implement to pass tests
#   ./tdd.sh refactor               # Refactor while keeping tests green
#   ./tdd.sh ship  <spec-file>      # Commit, create PR, archive spec
#   ./tdd.sh full  <spec-file>      # Run all four phases sequentially
#   ./tdd.sh watch [phase] [--resolve]  # Live-tail or summarize a phase log
#
# This project has two test frameworks:
#   C++:    ctest (GTest), test files in tests/
#   Python: pytest, test files in python/tests/

set -euo pipefail

# --- Configuration ---
TEST_DIR="${TEST_DIR:-tests}"
PYTHON_TEST_DIR="python/tests"
SRC_DIR="${SRC_DIR:-src}"
PROMPT_DIR=".claude/prompts"
HOOK_DIR=".claude/hooks"

# Post-cycle PR settings
TDD_AUTO_MERGE="${TDD_AUTO_MERGE:-false}"
TDD_DELETE_BRANCH="${TDD_DELETE_BRANCH:-false}"
TDD_BASE_BRANCH="${TDD_BASE_BRANCH:-main}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# --- Helpers ---

find_test_files() {
  find . -type f \( \
    -name "test_*.cpp" -o \
    -name "test_*.py" -o \
    -name "*_test.cpp" -o \
    -name "*_test.py" \
  \) ! -path "*/build/*" ! -path "*/_deps/*" ! -path "*/.git/*" ! -path "*/__pycache__/*" ! -path "*/.venv/*" ! -path "*/venv/*"
}

lock_tests() {
  echo -e "${YELLOW}Locking test files...${NC}"
  local count=0
  while IFS= read -r f; do
    chmod 444 "$f"
    echo -e "   ${YELLOW}locked:${NC} $f"
    ((count++))
  done < <(find_test_files)

  # Lock test directories to prevent new file creation
  for d in "$TEST_DIR" "$PYTHON_TEST_DIR"; do
    if [[ -d "$d" ]]; then
      find "$d" -type d | while IFS= read -r dir; do
        chmod 555 "$dir"
      done
    fi
  done

  echo -e "   ${YELLOW}$count file(s) locked${NC}"
}

unlock_tests() {
  echo -e "${BLUE}Unlocking test files...${NC}"
  # Unlock directories first
  for d in "$TEST_DIR" "$PYTHON_TEST_DIR"; do
    if [[ -d "$d" ]]; then
      find "$d" -type d -exec chmod 755 {} \; 2>/dev/null || true
    fi
  done

  local count=0
  while IFS= read -r f; do
    chmod 644 "$f"
    ((count++))
  done < <(find_test_files)
  echo -e "   ${BLUE}$count file(s) unlocked${NC}"
}

ensure_hooks_executable() {
  if [[ -f "$HOOK_DIR/pre-tool-use.sh" ]]; then
    chmod +x "$HOOK_DIR/pre-tool-use.sh"
  fi
}

_phase_summary() {
  # Extract the final agent message from the stream-json log and print a
  # compact summary.  This keeps the orchestrator's context window lean —
  # the full transcript stays on disk at /tmp/tdd-{phase}.log.
  local phase="$1"
  local exit_code="$2"
  local log="/tmp/tdd-${phase}.log"

  local summary
  summary=$(tail -20 "$log" 2>/dev/null | python3 -c "
import json, sys
texts = []
for line in sys.stdin:
    try:
        d = json.loads(line.strip())
        if d.get('type') == 'assistant':
            for c in d.get('message', {}).get('content', []):
                if c.get('type') == 'text':
                    texts.append(c['text'])
    except Exception:
        pass
if texts:
    print(texts[-1][:500])
" 2>/dev/null)

  if [[ -n "$summary" ]]; then
    echo -e "${YELLOW}[${phase}]${NC} $summary"
  fi
  echo -e "${YELLOW}[${phase}]${NC} Phase complete (exit: $exit_code). Log: $log"
  return "$exit_code"
}

# --- Phase Runners ---

run_red() {
  local spec_file="${1:?Usage: tdd.sh red <spec-file>}"

  if [[ ! -f "$spec_file" ]]; then
    echo -e "${RED}Error: Spec file not found: $spec_file${NC}" >&2
    exit 1
  fi

  echo ""
  echo -e "${RED}======================================================${NC}"
  echo -e "${RED}  TDD RED PHASE -- Writing Failing Tests${NC}"
  echo -e "${RED}======================================================${NC}"
  echo -e "  Spec:        $spec_file"
  echo -e "  C++ tests:   $TEST_DIR/ (GTest/ctest)"
  echo -e "  Python tests: $PYTHON_TEST_DIR/ (pytest)"
  echo ""

  export TDD_PHASE="red"

  local exit_code=0
  claude \
    --output-format stream-json \
    --append-system-prompt "$(cat "$PROMPT_DIR/tdd-red.md")

## Context
- Design spec path: $spec_file
- C++ test directory: $TEST_DIR
- Python test directory: $PYTHON_TEST_DIR
- C++ test framework: GTest (built via cmake, run via ctest)
- Python test framework: pytest (run via uv)
- Existing test files: $(find_test_files | tr '\n' ', ' || echo 'none')

Read the spec file first, then write your tests." \
    --allowed-tools "Read,Write,Edit,Bash" \
    -p "Read the spec file first, then write your tests." \
    > /tmp/tdd-red.log 2>&1 || exit_code=$?

  _phase_summary "red" "$exit_code"
}

run_green() {
  # Verify tests exist
  local test_count
  test_count=$(find_test_files | wc -l | tr -d ' ')
  if [[ "$test_count" -eq 0 ]]; then
    echo -e "${RED}Error: No test files found. Run 'tdd.sh red <spec>' first.${NC}" >&2
    exit 1
  fi

  echo ""
  echo -e "${GREEN}======================================================${NC}"
  echo -e "${GREEN}  TDD GREEN PHASE -- Implementing to Pass${NC}"
  echo -e "${GREEN}======================================================${NC}"
  echo -e "  Source dir:   $SRC_DIR"
  echo -e "  C++ tests:    $TEST_DIR/ (ctest)"
  echo -e "  Python tests: $PYTHON_TEST_DIR/ (pytest)"
  echo -e "  Test files:   $test_count file(s)"
  echo ""

  # OS-level enforcement
  lock_tests
  ensure_hooks_executable

  export TDD_PHASE="green"

  # Run agent (unlock on exit regardless of success/failure)
  trap unlock_tests EXIT

  local exit_code=0
  claude \
    --output-format stream-json \
    --append-system-prompt "$(cat "$PROMPT_DIR/tdd-green.md")

## Context
- Source directory: $SRC_DIR
- C++ test directory: $TEST_DIR (READ-ONLY -- do not attempt to modify)
- Python test directory: $PYTHON_TEST_DIR (READ-ONLY -- do not attempt to modify)
- C++ build: cd build && cmake --build . && ctest
- Python test: cd build && PYTHONPATH=.:../python uv run --with pytest --with gymnasium --with numpy pytest ../python/tests/
- Test files: $(find_test_files | tr '\n' ', ')

Start by reading the test files to understand what's expected, then implement iteratively." \
    --allowed-tools "Read,Write,Edit,Bash" \
    -p "Read the test files to understand what's expected, then implement iteratively." \
    > /tmp/tdd-green.log 2>&1 || exit_code=$?

  _phase_summary "green" "$exit_code"
}

run_refactor() {
  echo ""
  echo -e "${BLUE}======================================================${NC}"
  echo -e "${BLUE}  TDD REFACTOR PHASE -- Improving Quality${NC}"
  echo -e "${BLUE}======================================================${NC}"
  echo -e "  C++ tests:    ctest"
  echo -e "  Python tests: pytest"
  echo ""

  # Ensure tests are unlocked (refactor may touch test readability)
  unlock_tests 2>/dev/null || true

  export TDD_PHASE="refactor"

  local exit_code=0
  claude \
    --output-format stream-json \
    --append-system-prompt "$(cat "$PROMPT_DIR/tdd-refactor.md")

## Context
- Source directory: $SRC_DIR
- C++ test directory: $TEST_DIR
- Python test directory: $PYTHON_TEST_DIR
- C++ build/test: cd build && cmake --build . && ctest
- Python test: cd build && PYTHONPATH=.:../python uv run --with pytest --with gymnasium --with numpy pytest ../python/tests/

Start by running the full test suite to confirm your green baseline, then refactor." \
    --allowed-tools "Read,Write,Edit,Bash" \
    -p "Run the full test suite to confirm your green baseline, then refactor." \
    > /tmp/tdd-refactor.log 2>&1 || exit_code=$?

  _phase_summary "refactor" "$exit_code"
}

run_ship() {
  local spec_file="${1:?Usage: tdd.sh ship <spec-file>}"
  local feature_name
  feature_name="$(basename "$spec_file" .md)"
  local branch="tdd/${feature_name}"

  echo ""
  echo -e "${YELLOW}======================================================${NC}"
  echo -e "${YELLOW}  SHIPPING -- commit, PR, archive spec${NC}"
  echo -e "${YELLOW}======================================================${NC}"
  echo ""

  # Create feature branch
  git checkout -b "$branch" 2>/dev/null || git checkout "$branch"

  # Stage all changes and commit
  git add -A
  git commit -m "feat(${feature_name}): implement via TDD cycle

Spec: ${spec_file}
Red-green-refactor complete. All tests passing."

  # Push and create PR
  git push -u origin "$branch"

  local pr_url
  pr_url=$(gh pr create \
    --base "$TDD_BASE_BRANCH" \
    --title "feat(${feature_name}): TDD cycle complete" \
    --body "## TDD Cycle: ${feature_name}

**Spec:** \`${spec_file}\`

### Phases completed
- [x] RED — failing tests written
- [x] GREEN — implementation passes all tests
- [x] REFACTOR — code quality improved

### Spec archived
The spec file has been deleted from the working tree. It is preserved in this branch's git history.

---
*Generated by [claude-tdd-kit](https://github.com/kurtbell87/claude-tdd-kit)*")

  echo -e "  ${GREEN}PR created:${NC} $pr_url"

  # Auto-merge if configured
  if [[ "$TDD_AUTO_MERGE" == "true" ]]; then
    echo -e "  ${YELLOW}Auto-merging...${NC}"
    gh pr merge "$pr_url" --merge
    echo -e "  ${GREEN}Merged.${NC}"

    # Return to base branch
    git checkout "$TDD_BASE_BRANCH"
    git pull

    # Delete branch if configured
    if [[ "$TDD_DELETE_BRANCH" == "true" ]]; then
      git branch -d "$branch" 2>/dev/null || true
      gh api -X DELETE "repos/{owner}/{repo}/git/refs/heads/${branch}" 2>/dev/null || true
      echo -e "  ${GREEN}Branch deleted.${NC}"
    fi
  fi

  # Archive: delete the spec file (it's preserved in git history)
  if [[ -f "$spec_file" ]]; then
    rm "$spec_file"
    git add "$spec_file"
    git commit -m "chore: archive spec ${spec_file}"
    git push
    echo -e "  ${GREEN}Spec archived:${NC} $spec_file removed (preserved in git history)"
  fi

  echo ""
  echo -e "${GREEN}======================================================${NC}"
  echo -e "${GREEN}  Shipped! PR: $pr_url${NC}"
  echo -e "${GREEN}======================================================${NC}"
}

run_full() {
  local spec_file="${1:?Usage: tdd.sh full <spec-file>}"

  echo -e "${YELLOW}Running full TDD cycle: RED -> GREEN -> REFACTOR -> SHIP${NC}"
  echo ""

  run_red "$spec_file"

  echo ""
  echo -e "${YELLOW}--- Red phase complete. Starting green phase... ---${NC}"
  echo ""

  run_green

  echo ""
  echo -e "${YELLOW}--- Green phase complete. Starting refactor phase... ---${NC}"
  echo ""

  run_refactor

  echo ""
  echo -e "${YELLOW}--- Refactor phase complete. Shipping... ---${NC}"
  echo ""

  run_ship "$spec_file"

  echo ""
  echo -e "${YELLOW}======================================================${NC}"
  echo -e "${YELLOW}  Full TDD cycle complete!${NC}"
  echo -e "${YELLOW}======================================================${NC}"
}

# --- Main ---

case "${1:-help}" in
  red)      shift; run_red "$@" ;;
  green)    run_green ;;
  refactor) run_refactor ;;
  ship)     shift; run_ship "$@" ;;
  full)     shift; run_full "$@" ;;
  watch)    shift; exec uv run python3 scripts/tdd-watch.py "$@" ;;
  help|*)
    echo "Usage: tdd.sh <phase> [args]"
    echo ""
    echo "Phases:"
    echo "  red   <spec-file>   Write failing tests from design spec"
    echo "  green               Implement minimum code to pass tests"
    echo "  refactor            Refactor while keeping tests green"
    echo "  ship  <spec-file>   Commit, create PR, archive spec"
    echo "  full  <spec-file>   Run all four phases (red -> green -> refactor -> ship)"
    echo "  watch [phase]       Live-tail a running phase (--resolve for summary)"
    echo ""
    echo "Environment:"
    echo "  TEST_DIR=tests              C++ test directory (default: tests)"
    echo "  SRC_DIR=src                 Source directory (default: src)"
    echo "  TDD_AUTO_MERGE='false'      Auto-merge PR after creation"
    echo "  TDD_DELETE_BRANCH='false'   Delete feature branch after merge"
    echo "  TDD_BASE_BRANCH='main'      Base branch for PRs"
    ;;
esac
