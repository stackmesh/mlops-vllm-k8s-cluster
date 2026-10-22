#!/usr/bin/env bash
# Build the gateway image, load it into a kind cluster, and apply the lab.
set -euo pipefail

name="${KIND_CLUSTER:-mlops-lab}"

if ! command -v kind >/dev/null || ! command -v kubectl >/dev/null; then
  echo "kind and kubectl are required" >&2
  exit 1
fi

docker build -t mlops-gateway:local .
kind create cluster --name "$name" || true
kind load docker-image mlops-gateway:local --name "$name"
kubectl apply -k k8s/base
kubectl -n inference rollout status deploy/gateway --timeout=180s
echo "Gateway service: kubectl -n inference port-forward svc/gateway 8080:8080"
