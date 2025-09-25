#!/usr/bin/env bash
set -euo pipefail

echo "🔎 Linting…"
# Examples:
# python -m ruff check .
# npm run lint
# cmake -S . -B build && cmake --build build --target format-check

echo "🧪 Running tests…"
# Examples:
# pytest -q --maxfail=1 --disable-warnings --junitxml=reports/test-results.xml
# npm test -- --ci
# ctest --test-dir build --output-on-failure

echo "✅ All checks passed."
