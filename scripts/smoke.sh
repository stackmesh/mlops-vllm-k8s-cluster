#!/usr/bin/env bash
set -euo pipefail

base="${1:-http://127.0.0.1:8080}"

curl --fail --silent --show-error "$base/health"
echo
curl --fail --silent --show-error "$base/v1/budget" >/dev/null
curl --fail --silent --show-error \
  -H 'content-type: application/json' \
  -d '{"model":"qwen2.5-7b-instruct","quant":"awq","replicas":2}' \
  "$base/v1/plans"
echo
curl --fail --silent --show-error \
  -H 'content-type: application/json' \
  -d '{"model":"local","messages":[{"role":"user","content":"How does the resource quota cap physical GPU requests?"}]}' \
  "$base/v1/chat/completions"
echo
