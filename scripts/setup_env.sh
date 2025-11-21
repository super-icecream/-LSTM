#!/bin/bash
# Thin wrapper to run the canonical environment setup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
exec "${PROJECT_ROOT}/environment/setup_env.sh" "$@"

