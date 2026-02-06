#!/usr/bin/env bash
# tdd-aliases.sh -- Source this in your shell
#
#   source /path/to/tdd-aliases.sh
#
# Then use:
#   tdd-red docs/my-feature.md
#   tdd-green
#   tdd-refactor
#   tdd-full docs/my-feature.md
#   tdd-watch green
#   tdd-status

TDD_SCRIPT="./tdd.sh"

alias tdd-red='bash $TDD_SCRIPT red'
alias tdd-green='bash $TDD_SCRIPT green'
alias tdd-refactor='bash $TDD_SCRIPT refactor'
alias tdd-full='bash $TDD_SCRIPT full'
alias tdd-watch='bash $TDD_SCRIPT watch'

tdd-status() {
  echo "TDD Status"
  echo "==========="
  echo ""
  echo "Phase: ${TDD_PHASE:-not set}"
  echo ""
  echo "Test files:"
  find . -type f \( -name "test_*.cpp" -o -name "test_*.py" -o -name "*_test.cpp" -o -name "*_test.py" \) \
    ! -path "*/build/*" ! -path "*/_deps/*" ! -path "*/.git/*" | while read -r f; do
    if [[ ! -w "$f" ]]; then
      echo "  LOCKED  $f"
    else
      echo "  open    $f"
    fi
  done
  echo ""
  echo "Run 'tdd-red <spec>' to start a new cycle."
}

tdd-unlock() {
  echo "Emergency unlock -- restoring write permissions on all test files..."
  find . -type f \( -name "test_*.cpp" -o -name "test_*.py" -o -name "*_test.cpp" -o -name "*_test.py" \) \
    ! -path "*/build/*" ! -path "*/_deps/*" ! -path "*/.git/*" -exec chmod 644 {} \;
  find tests -type d 2>/dev/null -exec chmod 755 {} \;
  find python/tests -type d 2>/dev/null -exec chmod 755 {} \;
  echo "Done."
}
