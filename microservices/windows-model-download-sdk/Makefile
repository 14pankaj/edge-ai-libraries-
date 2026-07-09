# SPDX-FileCopyrightText: (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

SHELL := /bin/bash

SDK_ROOT := $(abspath .)
MODEL_DOWNLOAD_DIR := $(abspath ../model-download)
COMPOSE_FILE := $(MODEL_DOWNLOAD_DIR)/docker/compose.yaml
VENV_DIR ?= .venv
PYTHON ?= python3
REGISTRY ?="intel/"
TAG ?= latest
ENABLED_PLUGINS ?= all
MODEL_PATH ?= $(HOME)/models
GETI_HOST ?=
GETI_TOKEN ?=
GETI_WORKSPACE_ID ?=
HLS_3D_POSE_CHECKPOINT_URL ?=
HLS_ECG_BASE_URL ?=
HLS_RPPG_MODEL_URL ?=
HUGGINGFACEHUB_API_TOKEN ?=
HEALTH_URL ?= http://localhost:8200/health
HEALTH_RETRIES ?= 60
HEALTH_INTERVAL ?= 2

.PHONY: help bootstrap backend-up backend-wait backend-down backend-logs venv install health test

help:
	@echo "Windows Model Download SDK helper targets"
	@echo "  make bootstrap    - Start backend, set up venv, install SDK, run health check"
	@echo "  make backend-up   - Start model-download backend via docker compose"
	@echo "  make backend-wait - Wait until backend health endpoint is ready"
	@echo "  make backend-down - Stop model-download backend"
	@echo "  make backend-logs - Tail backend logs"
	@echo "  make venv         - Create Python virtual environment"
	@echo "  make install      - Install SDK in editable mode"
	@echo "  make health       - Run SDK health check"
	@echo "  make test         - Run pytest suite"

bootstrap: backend-up install backend-wait health

backend-up:
	@REGISTRY="$(REGISTRY)" TAG="$(TAG)" ENABLED_PLUGINS="$(ENABLED_PLUGINS)" MODEL_PATH="$(MODEL_PATH)" GETI_HOST="$(GETI_HOST)" GETI_TOKEN="$(GETI_TOKEN)" GETI_WORKSPACE_ID="$(GETI_WORKSPACE_ID)" HLS_3D_POSE_CHECKPOINT_URL="$(HLS_3D_POSE_CHECKPOINT_URL)" HLS_ECG_BASE_URL="$(HLS_ECG_BASE_URL)" HLS_RPPG_MODEL_URL="$(HLS_RPPG_MODEL_URL)" HUGGINGFACEHUB_API_TOKEN="$(HUGGINGFACEHUB_API_TOKEN)" docker compose -f "$(COMPOSE_FILE)" up -d --build

backend-wait:
	@echo "Waiting for backend health at $(HEALTH_URL)"
	@for i in $$(seq 1 $(HEALTH_RETRIES)); do \
		if curl -fsS "$(HEALTH_URL)" >/dev/null 2>&1; then \
			echo "Backend is healthy"; \
			exit 0; \
		fi; \
		echo "[$$i/$(HEALTH_RETRIES)] backend not ready yet, retrying in $(HEALTH_INTERVAL)s"; \
		sleep "$(HEALTH_INTERVAL)"; \
	done; \
	echo "ERROR: Backend did not become healthy at $(HEALTH_URL)" >&2; \
	exit 1

backend-down:
	@REGISTRY="$(REGISTRY)" TAG="$(TAG)" ENABLED_PLUGINS="$(ENABLED_PLUGINS)" MODEL_PATH="$(MODEL_PATH)" GETI_HOST="$(GETI_HOST)" GETI_TOKEN="$(GETI_TOKEN)" GETI_WORKSPACE_ID="$(GETI_WORKSPACE_ID)" HLS_3D_POSE_CHECKPOINT_URL="$(HLS_3D_POSE_CHECKPOINT_URL)" HLS_ECG_BASE_URL="$(HLS_ECG_BASE_URL)" HLS_RPPG_MODEL_URL="$(HLS_RPPG_MODEL_URL)" HUGGINGFACEHUB_API_TOKEN="$(HUGGINGFACEHUB_API_TOKEN)" docker compose -f "$(COMPOSE_FILE)" down

backend-logs:
	@REGISTRY="$(REGISTRY)" TAG="$(TAG)" ENABLED_PLUGINS="$(ENABLED_PLUGINS)" MODEL_PATH="$(MODEL_PATH)" GETI_HOST="$(GETI_HOST)" GETI_TOKEN="$(GETI_TOKEN)" GETI_WORKSPACE_ID="$(GETI_WORKSPACE_ID)" HLS_3D_POSE_CHECKPOINT_URL="$(HLS_3D_POSE_CHECKPOINT_URL)" HLS_ECG_BASE_URL="$(HLS_ECG_BASE_URL)" HLS_RPPG_MODEL_URL="$(HLS_RPPG_MODEL_URL)" HUGGINGFACEHUB_API_TOKEN="$(HUGGINGFACEHUB_API_TOKEN)" docker compose -f "$(COMPOSE_FILE)" logs -f model_download_service

venv:
	@$(PYTHON) -m venv "$(VENV_DIR)"

install: venv
	@"$(VENV_DIR)/bin/python" -m pip install --upgrade pip
	@"$(VENV_DIR)/bin/python" -m pip install -e .

health:
	@"$(VENV_DIR)/bin/python" -c "import asyncio; from model_download_sdk.client import ModelDownloadSDK; client = ModelDownloadSDK(); print(asyncio.run(client.health_check())); asyncio.run(client.close())"

test:
	@"$(VENV_DIR)/bin/python" -m pytest -q
