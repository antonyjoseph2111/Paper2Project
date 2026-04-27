# Paper2Project Architecture

## 1. System objective

Paper2Project transforms a machine learning paper into an editable engineering artifact set:

- Structured paper understanding
- Executable ML pipeline plan
- User-editable decision config
- Modular PyTorch project files
- Runnable Google Colab notebook

The system optimizes for baseline reproducibility, transparency, and user control rather than pretending it can perfectly reverse-engineer every paper.

## 2. End-to-end workflow

```text
PDF Upload
  -> PDF Parser
  -> Parsed Paper JSON
  -> Paper Analyst Agent
  -> Analysis JSON
  -> Planner Agent
  -> Pipeline Plan JSON + Assumptions
  -> Decision Agent
  -> Editable Decision Config
  -> User Approval / Edits
  -> Code Generator Agent
  -> Project Files
  -> Notebook Builder Agent
  -> Colab Notebook
  -> Validation + Artifact Bundle
```

## 3. Agent responsibilities

### 3.1 Paper Analyst Agent

Purpose:

- Convert cleaned paper content into structured ML understanding.

Inputs:

- Parsed paper JSON
- Chunked methodology and model text

Outputs:

- Task type
- Domain
- Model type
- Components
- Training details
- Loss
- Metrics
- Confidence and ambiguity flags

Failure behavior:

- Returns partial JSON with `unknown`, `assumed`, and `confidence` annotations

### 3.2 Planner Agent

Purpose:

- Convert paper understanding into an executable ML pipeline.

Inputs:

- Paper analysis JSON
- Dataset suggestions

Outputs:

- Ordered execution steps
- Dataset requirements
- Model structure
- Training hyperparameters
- Assumptions list
- Open questions

Failure behavior:

- Uses defaults while preserving assumption tags

### 3.3 Decision Agent

Purpose:

- Present a human-editable control surface before generation.

Inputs:

- Pipeline plan
- Ambiguities
- Dataset alternatives

Outputs:

- Editable decision JSON
- Required approvals
- Recommended defaults

Failure behavior:

- Blocks code generation only for critical missing decisions
- Allows provisional defaults for non-critical items

### 3.4 Code Generator Agent

Purpose:

- Produce runnable, modular PyTorch project code.

Inputs:

- Approved decision config
- Pipeline plan
- Paper references

Outputs:

- `model.py`
- `data_loader.py`
- `train.py`
- `config.yaml`
- `README.md` for generated project

Failure behavior:

- Generates simplest valid baseline when paper details are incomplete

### 3.5 Notebook Builder Agent

Purpose:

- Build a runnable Colab notebook around the same approved project configuration.

Inputs:

- Approved config
- Generated code artifacts

Outputs:

- `paper2project_notebook.ipynb`

Failure behavior:

- Falls back to inline notebook code if file import strategy is unavailable

## 4. Agent communication contract

All agents exchange structured JSON only.

Rules:

- Every output includes `status`
- Every output includes `assumptions`
- Every uncertain field can carry:
  - `value`
  - `confidence`
  - `source_section`
  - `assumed`

Example:

```json
{
  "loss": {
    "value": "cross_entropy",
    "confidence": 0.74,
    "source_section": "Method",
    "assumed": false
  }
}
```

## 5. Core data model

### Parsed Paper

```json
{
  "title": "string",
  "problem": "string",
  "abstract": "string",
  "introduction": "string",
  "methodology_text": "string",
  "model_description": "string",
  "equations": ["string"],
  "keywords": ["string"],
  "sections": [{"name": "string", "content": "string"}]
}
```

### Paper Analysis

```json
{
  "task": "classification",
  "domain": "NLP",
  "input_data_type": "text",
  "output_format": "label",
  "model_type": "Transformer",
  "components": ["embedding", "self-attention", "feedforward"],
  "loss": "cross_entropy",
  "metrics": ["accuracy"],
  "training_details": {
    "optimizer": "adamw",
    "scheduler": "linear",
    "epochs": 3,
    "batch_size": 32
  },
  "ambiguities": ["dataset unspecified"],
  "assumptions": []
}
```

### Pipeline Plan

```json
{
  "steps": [
    "load_dataset",
    "preprocess_data",
    "build_model",
    "train_model",
    "evaluate_model"
  ],
  "dataset_requirements": {
    "source": "huggingface",
    "candidate_datasets": ["ag_news"],
    "split_strategy": "train/validation/test"
  },
  "model_structure": {
    "encoder": "transformer_encoder",
    "head": "linear_classifier"
  },
  "hyperparameters": {
    "learning_rate": 2e-5,
    "epochs": 3,
    "batch_size": 32
  },
  "assumptions": ["Using AG News because the paper dataset is unavailable"]
}
```

### Decision Config

```json
{
  "dataset": {
    "selected": "ag_news",
    "alternatives": ["dbpedia_14", "yelp_review_full"],
    "editable": true
  },
  "model": {
    "selected": "distilbert-base-uncased",
    "reason": "baseline transformer choice",
    "editable": true
  },
  "training": {
    "epochs": 3,
    "batch_size": 32,
    "learning_rate": 2e-5,
    "seed": 42
  },
  "approval_required": true
}
```

## 6. PDF ingestion strategy

### Primary parser

- PyMuPDF for reliable text extraction and page access

### Optional structured parser

- Grobid for title, abstract, references, and section segmentation

### Preferred source override

- If arXiv LaTeX source is available, prefer it over raw PDF parsing for equations and section boundaries

### Cleaning steps

1. Remove references section and bibliography spillover
2. Remove page headers, footers, and duplicate whitespace
3. Preserve equations as raw text
4. Split into named sections
5. Chunk long sections for LLM processing

## 7. Dataset mapping strategy

Heuristics:

- NLP:
  - Hugging Face datasets first
- Computer vision:
  - CIFAR-10, MNIST, ImageNet-compatible subsets, Oxford Pets
- Tabular:
  - UCI or Kaggle alternatives when licensing permits
- Unknown:
  - Synthetic placeholder dataset with explicit warning

Every suggestion includes:

- Match rationale
- Licensing note
- Ease-of-Colab note

## 8. Reproducibility design

Required:

- Global seed in config
- Generated `config.yaml`
- Saved assumptions and ambiguities
- Hyperparameter logging
- Deterministic DataLoader options where feasible

Optional:

- TensorBoard
- Weights & Biases
- Artifact manifest JSON

## 9. Reliability mechanisms

### Graceful degradation

- Missing methodology -> generate simpler baseline
- Missing dataset -> propose substitutes
- Ambiguous loss -> select standard loss for inferred task
- Missing hyperparameters -> use domain defaults

### Validation gates

- JSON schema validation at every step
- Required keys enforced by Pydantic
- Notebook generation only after approved config exists
- Execution-oriented checks on generated files

### Observability

- Job-level IDs
- Per-agent timings
- Assumption counters
- Parse confidence
- Generation success metrics

## 10. API design

Recommended endpoints:

- `POST /jobs`
- `GET /jobs/{job_id}`
- `GET /jobs/{job_id}/decision`
- `PATCH /jobs/{job_id}/decision`
- `POST /jobs/{job_id}/approve`
- `GET /jobs/{job_id}/artifacts`

## 11. Deployment model

Recommended production layout:

- FastAPI app
- Background worker for long jobs
- Object storage for PDFs and artifacts
- Postgres for job state
- Redis for queue and caching
- Provider abstraction for LLMs

## 12. Industry-fit rationale

This design is closer to how real ML engineering teams work:

- Papers are translated into baseline systems, not blindly replicated
- Dataset substitutions are explicit
- Project outputs are editable
- Notebook generation is operational, not decorative
- User approval prevents silent bad assumptions
