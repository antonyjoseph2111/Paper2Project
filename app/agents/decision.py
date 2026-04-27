from __future__ import annotations

from app.agents.base import Agent
from app.models.schemas import (
    DatasetDecision,
    DecisionConfig,
    JobRecord,
    ModelDecision,
    PipelinePlan,
    TrainingDecision,
)


class DecisionAgent(Agent):
    name = "decision"

    def run(self, job: JobRecord, plan: PipelinePlan) -> DecisionConfig:
        payload = {
            "plan": plan.model_dump(),
            "required_output_schema": {
                "dataset": {"selected": "ag_news", "source": "huggingface", "alternatives": ["dbpedia_14"], "editable": True, "reason": "Rationale"},
                "model": {"selected": "distilbert-base-uncased", "reason": "Rationale", "editable": True},
                "training": {
                    "epochs": 3,
                    "batch_size": 32,
                    "learning_rate": 2e-5,
                    "seed": 42,
                    "optimizer": "adamw",
                    "scheduler": "linear",
                    "loss": "cross_entropy",
                    "weight_decay": 0.01,
                    "max_length": 128,
                    "use_tensorboard": True,
                    "use_wandb": False
                },
                "approval_required": True,
                "assumptions": [],
                "unresolved_questions": []
            },
        }
        response = self.run_llm_json(job, payload)
        if response:
            try:
                return DecisionConfig.model_validate(response)
            except Exception:
                pass

        dataset_candidates = plan.dataset_requirements.candidate_datasets or [plan.dataset_requirements.synthetic_fallback]
        selected_dataset = dataset_candidates[0]
        return DecisionConfig(
            dataset=DatasetDecision(
                selected=selected_dataset,
                source=plan.dataset_requirements.source,
                alternatives=dataset_candidates[1:],
                reason="Planner-selected best effort dataset.",
            ),
            model=ModelDecision(
                selected=plan.model_structure.backbone or "baseline_model",
                reason="Planner-selected baseline model family.",
            ),
            training=TrainingDecision(
                epochs=int(plan.hyperparameters.get("epochs", 3)),
                batch_size=int(plan.hyperparameters.get("batch_size", 32)),
                learning_rate=float(plan.hyperparameters.get("learning_rate", 2e-5)),
                seed=int(plan.hyperparameters.get("seed", 42)),
                optimizer=str(plan.hyperparameters.get("optimizer", "adamw")),
                scheduler=str(plan.hyperparameters.get("scheduler", "linear")),
                loss=str(plan.hyperparameters.get("loss", "cross_entropy")),
                weight_decay=float(plan.hyperparameters.get("weight_decay", 0.0)),
                max_length=int(plan.hyperparameters.get("max_length", 128)),
                use_tensorboard=bool(plan.hyperparameters.get("use_tensorboard", True)),
                use_wandb=bool(plan.hyperparameters.get("use_wandb", False)),
            ),
            assumptions=plan.assumptions,
            unresolved_questions=plan.open_questions,
        )
