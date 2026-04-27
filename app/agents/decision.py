from app.agents.base import Agent
from app.models.schemas import DecisionConfig, DatasetDecision, ModelDecision, PipelinePlan, TrainingDecision


class DecisionAgent(Agent):
    name = "decision"

    def run(self, plan: PipelinePlan) -> DecisionConfig:
        dataset = plan.dataset_requirements.candidate_datasets[0] if plan.dataset_requirements.candidate_datasets else "synthetic_placeholder"
        model = plan.model_structure.backbone or "mlp_baseline"
        return DecisionConfig(
            dataset=DatasetDecision(
                selected=dataset,
                alternatives=plan.dataset_requirements.candidate_datasets[1:],
                reason="Best-effort match from planner heuristics.",
            ),
            model=ModelDecision(
                selected=model,
                reason="Planner-selected baseline architecture.",
            ),
            training=TrainingDecision(**{
                "epochs": int(plan.hyperparameters.get("epochs", 3)),
                "batch_size": int(plan.hyperparameters.get("batch_size", 32)),
                "learning_rate": float(plan.hyperparameters.get("learning_rate", 2e-5)),
                "seed": int(plan.hyperparameters.get("seed", 42)),
            }),
            assumptions=plan.assumptions,
            unresolved_questions=plan.open_questions,
        )
