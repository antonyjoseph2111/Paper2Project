from __future__ import annotations

from app.agents.base import Agent
from app.models.schemas import EvidenceField, JobRecord, PaperAnalysis, ParsedPaper, TrainingDetails
from app.services.fallbacks import heuristic_analysis


class PaperAnalystAgent(Agent):
    name = "paper_analyst"

    def run(self, job: JobRecord, parsed_paper: ParsedPaper) -> PaperAnalysis:
        payload = {
            "title": parsed_paper.title,
            "abstract": parsed_paper.abstract,
            "introduction": parsed_paper.introduction,
            "methodology_text": parsed_paper.methodology_text,
            "model_description": parsed_paper.model_description,
            "equations": parsed_paper.equations[:10],
            "keywords": parsed_paper.keywords,
            "sections": [section.model_dump() for section in parsed_paper.sections[:12]],
            "output_schema_example": {
                "task": {"value": "classification", "confidence": 0.91, "source_section": "abstract", "assumed": False},
                "domain": {"value": "NLP", "confidence": 0.9, "source_section": "methodology", "assumed": False},
                "input_data_type": {"value": "text", "confidence": 0.9, "source_section": "methodology", "assumed": False},
                "output_format": {"value": "label", "confidence": 0.8, "source_section": "experiments", "assumed": True},
                "model_type": {"value": "transformer", "confidence": 0.87, "source_section": "methodology", "assumed": False},
                "components": {"value": ["embedding", "self_attention"], "confidence": 0.85, "source_section": "methodology", "assumed": False},
                "loss": {"value": "cross_entropy", "confidence": 0.7, "source_section": "training", "assumed": True},
                "metrics": {"value": ["accuracy"], "confidence": 0.75, "source_section": "results", "assumed": False},
                "training_details": {
                    "optimizer": "adamw",
                    "scheduler": "linear",
                    "epochs": 3,
                    "batch_size": 32,
                    "learning_rate": 2e-5,
                    "weight_decay": 0.01,
                    "max_length": 128
                },
                "ambiguities": [],
                "assumptions": []
            },
        }
        response = self.run_llm_json(job, payload)
        if response:
            try:
                return PaperAnalysis.model_validate(response)
            except Exception:
                pass
        fallback = heuristic_analysis(
            f"{parsed_paper.abstract}\n{parsed_paper.introduction}\n{parsed_paper.methodology_text}\n{parsed_paper.model_description}"
        )
        if not fallback.training_details:
            fallback.training_details = TrainingDetails()
        return fallback
