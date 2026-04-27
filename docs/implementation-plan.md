# Implementation Plan

## Phase 1: Foundation

Goal:

- Establish typed contracts and deterministic orchestration.

Work:

1. Create Pydantic schemas for every agent boundary
2. Implement PDF parsing and cleaning
3. Create prompt templates
4. Create a stubbed orchestration flow
5. Generate artifact directories per job

Exit criteria:

- Uploading a PDF produces parsed JSON and a decision config

## Phase 2: Agent execution

Goal:

- Add LLM-backed structured extraction and planning.

Work:

1. Implement provider-agnostic LLM adapter
2. Add Paper Analyst prompt and parser
3. Add Planner prompt and parser
4. Add Decision Agent prompt and editable config generation
5. Add retries and schema validation

Exit criteria:

- A paper consistently produces valid analysis and plan JSON

## Phase 3: Code and notebook generation

Goal:

- Produce runnable baseline artifacts.

Work:

1. Implement PyTorch file generators
2. Implement config synthesis
3. Implement Colab notebook generation with `nbformat`
4. Add artifact manifest
5. Add simple smoke validation on generated files

Exit criteria:

- System writes project files and notebook without syntax errors

## Phase 4: Reliability and operations

Goal:

- Make the system service-grade.

Work:

1. Add persistent job storage
2. Move jobs to a background queue
3. Add metrics and structured logging
4. Add code execution sandbox for validation
5. Add provider fallback policies

Exit criteria:

- Jobs survive restarts and expose execution telemetry

## Phase 5: Evaluation

Goal:

- Measure whether outputs are useful.

Work:

1. Build a benchmark set of papers
2. Track parse quality
3. Track JSON validity rate
4. Track code execution success rate
5. Track notebook run success rate

Exit criteria:

- System quality is measurable and improving release over release

## Initial backlog

1. Replace in-memory store with Postgres
2. Add arXiv source fetcher
3. Add Grobid integration
4. Add unit tests for agent outputs
5. Add notebook smoke test runner
6. Add dataset recommendation ranking
