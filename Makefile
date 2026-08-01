PYTHON := .venv/bin/python

.PHONY: dev test test-contract test-provider-contracts test-agent-contracts test-core test-providers test-examples test-agents test-docs test-release test-cov lint typecheck smoke smoke-agents compile support-matrix-check security-check check build release-install-check release-evidence release-check clean

dev:
	uv venv .venv
	uv pip install -e '.[dev]'

test:
	$(PYTHON) -m pytest -q

test-contract:
	$(PYTHON) -m pytest tests/test_public_contract.py tests/test_api_stability.py tests/test_provider_support.py tests/test_agent_capabilities.py tests/contracts/test_tier1_provider_contracts.py -q

test-provider-contracts:
	$(PYTHON) -m pytest tests/contracts/test_tier1_provider_contracts.py -q

test-agent-contracts:
	$(PYTHON) -m pytest tests/contracts/test_agent_runtime_contracts.py -q

test-core:
	$(PYTHON) -m pytest tests/test_core.py tests/test_gateway.py tests/test_runtime.py tests/test_transport.py tests/test_http.py tests/test_catalog_and_middleware.py -q

test-providers:
	$(PYTHON) -m pytest tests/test_openai_provider.py tests/test_anthropic_provider.py tests/test_azure_openai_provider.py tests/test_gemini_provider.py tests/test_vllm_provider.py tests/test_bedrock_provider.py tests/test_deepseek_provider.py tests/test_kimi_provider.py tests/test_ollama_provider.py tests/test_qwen_provider.py tests/test_hosted_tools.py tests/test_realtime.py -q

test-examples:
	$(PYTHON) -m pytest tests/test_small_business_loan_example.py tests/test_hr_candidate_selection_example.py tests/test_workflow_examples.py tests/test_operations_hardening_example.py tests/test_production_examples.py -q

test-agents:
	$(PYTHON) -m pytest tests/test_agent.py tests/test_agent_dx.py tests/test_agent_extensions.py tests/test_agent_safety_runtime.py tests/test_tool_timeout_safety.py tests/test_postgres_agent_runtime.py tests/test_platform_parity.py tests/test_workflow.py tests/test_workflow_graph.py tests/test_workflow_state.py tests/test_workflow_adapters.py tests/test_skills.py tests/test_skill_packages.py tests/contracts/test_agent_runtime_contracts.py -q

test-docs:
	$(PYTHON) -m pytest tests/test_docs_onboarding.py tests/test_operations_docs.py -q

test-release:
	$(PYTHON) -m pytest tests/test_release_artifacts.py tests/test_live_smoke.py -q

test-cov:
	$(PYTHON) -m pytest --cov=src/zhivex_ai --cov-report=term-missing:skip-covered --cov-fail-under=80 -q

lint:
	$(PYTHON) -m ruff check src tests examples

typecheck:
	$(PYTHON) -m mypy

smoke:
	$(PYTHON) scripts/run_live_smoke.py

smoke-agents:
	ZHIVEX_SMOKE_AGENTS=1 ZHIVEX_SMOKE_STRICT=1 $(PYTHON) scripts/run_live_smoke.py

compile:
	$(PYTHON) -m compileall src tests examples

support-matrix-check:
	$(PYTHON) scripts/generate_support_matrix.py --check-readme

security-check:
	$(PYTHON) -m pip check
	$(PYTHON) -m pip_audit . --strict
	$(PYTHON) -m pip_audit --local --skip-editable

check: compile lint typecheck support-matrix-check test-cov

build:
	rm -rf dist build
	$(PYTHON) -m build --no-isolation --outdir dist

release-install-check:
	$(PYTHON) scripts/verify_release_artifacts.py --dist-dir dist

release-evidence:
	$(PYTHON) scripts/collect_release_evidence.py

release-check: check test-release build release-install-check security-check
	$(PYTHON) -m twine check dist/*

clean:
	rm -rf .venv dist build src/*.egg-info .pytest_cache
