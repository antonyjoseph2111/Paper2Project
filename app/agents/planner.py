from __future__ import annotations

from app.agents.base import Agent
from app.models.schemas import DatasetRequirements, JobRecord, ModelStructure, PaperAnalysis, PipelinePlan
from app.services.dataset_mapper import suggest_datasets


class PlannerAgent(Agent):
    name = "planner"

    def run(self, job: JobRecord, analysis: PaperAnalysis) -> PipelinePlan:
        datasets = suggest_datasets(analysis)
        payload = {
            "analysis": analysis.model_dump(),
            "dataset_suggestions": datasets,
            "required_output_schema": {
                "steps": ["load_dataset", "preprocess_data", "build_model", "train_model", "evaluate_model"],
                "dataset_requirements": {
                    "source": "huggingface",
                    "candidate_datasets": ["ag_news"],
                    "split_strategy": "train/validation/test",
                    "notes": "Dataset rationale",
                    "synthetic_fallback": "synthetic_text_classification"
                },
                "model_structure": {
                    "backbone": "distilbert-base-uncased",
                    "head": "linear_classifier",
                    "notes": "Implementation notes",
                    "reference_components": ["embedding", "encoder", "classifier_head"]
                },
                "hyperparameters": {"epochs": 3, "batch_size": 32, "learning_rate": 2e-5},
                "assumptions": [],
                "open_questions": [],
                "implementation_notes": []
            },
        }
        response = self.run_llm_json(job, payload)
        if response:
            try:
                return PipelinePlan.model_validate(response)
            except Exception:
                pass

        task = str(analysis.task.value).lower()
        domain = str(analysis.domain.value).lower()
        backbone = "distilbert-base-uncased"
        head = "linear_classifier"
        source = datasets["source"]
        if domain == "cv" and task == "classification":
            backbone = "resnet18"
        elif domain == "cv" and task == "segmentation":
            backbone = "unet"
            head = "segmentation_head"
        elif domain == "rl":
            backbone = "dqn_mlp"
            head = "q_value_head"
        elif domain == "tabular":
            backbone = "mlp"
            head = "regression_head" if task == "regression" else "classification_head"
        elif task == "generation":
            backbone = "gru_language_model"
            head = "lm_head"

        return PipelinePlan(
            steps=[
                "load_dataset",
                "preprocess_data",
                "build_model",
                "train_model",
                "evaluate_model",
                "export_artifacts",
            ],
            dataset_requirements=DatasetRequirements(
                source=source,
                candidate_datasets=datasets["candidates"],
                split_strategy="train/validation/test",
                notes=datasets["notes"],
                synthetic_fallback=datasets["synthetic_fallback"],
            ),
            model_structure=ModelStructure(
                backbone=backbone,
                head=head,
                notes="Fallback planner output due to unavailable or invalid LLM response.",
                reference_components=list(analysis.components.value),
            ),
            hyperparameters=analysis.training_details.model_dump(),
            assumptions=list(analysis.assumptions),
            open_questions=list(analysis.ambiguities),
            implementation_notes=["Fallback plan generated heuristically."],
        )
