<!--
SPDX-FileCopyrightText: (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Agent Quality Handler

The Agent Quality Handler is a standalone, configuration-driven service that
runs an Agentic Predictive Maintenance graph against detections from an
external storage API. It orchestrates Policy, Analysis, Evidence, and Ticketing
agents in rule-based fallback mode or with an OpenAI-compatible LLM.

Below, you'll find links to detailed documentation to help you get started,
configure, integrate, and deploy the microservice.

## Documentation

- **Overview**
  - [Overview](./docs/user-guide/index.md): A high-level introduction to the microservice and its capabilities.
  - [How It Works](./docs/user-guide/how-it-works.md): Architecture, data flow, run lifecycle, and external integrations.

- **Getting Started**
  - [Get Started](./docs/user-guide/get-started.md): Run the service in fallback, LLM, or development mode.
  - [System Requirements](./docs/user-guide/get-started/system-requirements.md): Hardware and software requirements for running the microservice.
  - [Build from Source](./docs/user-guide/build-from-source.md): Instructions for setting up and running the microservice from source code.

- **Integration and API**
  - [API Reference](./docs/user-guide/api-reference.md): REST endpoints, request and response formats, and metrics.
  - [Agent Service Integration Guide](./docs/user-guide/agent-service-integration-guide.md): Integrate the microservice with external detection, storage, and MQTT services.

- **Support**
  - [Troubleshooting](./docs/user-guide/troubleshooting.md): Solutions for common configuration and runtime issues.
  - [Release Notes](./docs/user-guide/release-notes.md): Information about updates, improvements, and bug fixes.
