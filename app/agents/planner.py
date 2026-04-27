from app.agents.base import Agent
from app.models.schemas import DatasetRequirements, ModelStructure, PaperAnalysis, PipelinePlan
from app.services.dataset_mapper import suggest_datasets


class PlannerAgent(Agent):
    name = "planner"

    def run(self, analysis: PaperAnalysis) -> PipelinePlan:
        datasets = suggest_datasets(analysis)
        assumptions = list(analysis.assumptions)
        if not datasets:
            datasets = ["synthetic_placeholder"]
            assumptions.append("No strong dataset match found; using a synthetic placeholder.")
        backbone = "distilbert-base-uncased" if analysis.domain == "NLP" else "mlp_baseline"
        head = "linear_classifier" if analysis.task == "classification" else "prediction_head"
        return PipelinePlan(
            steps=[
                "load_dataset",
                "preprocess_data",
                "build_model",
                "train_model",
                "evaluate_model",
            ],
            dataset_requirements=DatasetRequirements(
                source="huggingface" if analysis.domain == "NLP" else "custom",
                candidate_datasets=datasets,
                split_strategy="train/validation/test",
                notes="Top candidates are suggestions, not guaranteed matches.",
            ),
            model_structure=ModelStructure(
                backbone=backbone,
                head=head,
                notes="Backbone may be substituted by the Decision Agent before generation.",
            ),
            hyperparameters=analysis.training_details.model_dump(),
            assumptions=assumptions,
            open_questions=list(analysis.ambiguities),
        )
