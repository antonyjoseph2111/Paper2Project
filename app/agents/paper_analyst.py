from app.agents.base import Agent
from app.models.schemas import PaperAnalysis, ParsedPaper


class PaperAnalystAgent(Agent):
    name = "paper_analyst"

    def run(self, parsed_paper: ParsedPaper) -> PaperAnalysis:
        text = f"{parsed_paper.abstract}\n{parsed_paper.methodology_text}".lower()
        domain = "NLP" if "text" in text or "language" in text or "transformer" in text else "unknown"
        model_type = "Transformer" if "transformer" in text or "attention" in text else "unknown"
        task = "classification" if "classification" in text or "accuracy" in text else "unknown"
        input_data_type = "text" if domain == "NLP" else "unknown"
        output_format = "label" if task == "classification" else "unknown"
        components = ["embedding", "self-attention", "feedforward"] if model_type == "Transformer" else []
        loss = "cross_entropy" if task == "classification" else "unknown"
        metrics = ["accuracy"] if "accuracy" in text else []
        ambiguities = []
        assumptions = []
        if task == "unknown":
            ambiguities.append("Task type could not be confidently inferred from parsed text.")
        if model_type == "unknown":
            assumptions.append("Default model family may need user correction.")
        return PaperAnalysis(
            task=task,
            domain=domain,
            input_data_type=input_data_type,
            output_format=output_format,
            model_type=model_type,
            components=components,
            loss=loss,
            metrics=metrics,
            ambiguities=ambiguities,
            assumptions=assumptions,
        )
