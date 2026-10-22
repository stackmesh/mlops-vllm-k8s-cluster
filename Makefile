PYTHON ?= python3

.PHONY: install test compile plan render serve examples

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

compile:
	$(PYTHON) -m compileall -q src dags tests

plan:
	$(PYTHON) -m mlops_cluster.cli plan --model qwen2.5-7b-instruct --quant awq --replicas 2

examples:
	$(PYTHON) -m mlops_cluster.cli render --model qwen2.5-7b-instruct --quant awq --replicas 2 --out k8s/examples/qwen2.5-7b-awq
	$(PYTHON) -m mlops_cluster.cli render --model llama-3.1-8b-instruct --quant awq --replicas 1 --out k8s/examples/llama-3.1-8b-awq
	$(PYTHON) -m mlops_cluster.cli render --model llama-3.1-70b-instruct --quant awq --replicas 1 --out k8s/examples/llama-3.1-70b-awq

serve:
	$(PYTHON) -m mlops_cluster.cli serve --host 0.0.0.0 --port 8080
