from __future__ import annotations

from app.models.schemas import EvidenceField, PaperAnalysis, TrainingDetails


def heuristic_analysis(parsed_text: str) -> PaperAnalysis:
    text = parsed_text.lower()
    domain = "NLP"
    input_type = "text"
    task = "classification"
    model_type = "transformer"
    output = "label"
    loss = "cross_entropy"
    metrics = ["accuracy"]
    components = ["embedding", "encoder", "prediction_head"]
    assumptions: list[str] = []
    ambiguities: list[str] = []

    if any(token in text for token in ["image", "visual", "cnn", "resnet", "segmentation", "detection"]):
        domain = "CV"
        input_type = "image"
        model_type = "cnn" if "cnn" in text or "resnet" in text else "vision_transformer"
        metrics = ["accuracy"]
        components = ["backbone", "pooling", "classifier_head"]
    if any(token in text for token in ["reward", "policy", "mdp", "environment", "q-learning"]):
        domain = "RL"
        input_type = "state_vector"
        task = "reinforcement_learning"
        output = "action"
        model_type = "dqn"
        loss = "temporal_difference"
        metrics = ["episode_return"]
        components = ["q_network", "target_network", "replay_buffer"]
    elif any(token in text for token in ["regression", "mse", "mae", "rmse"]):
        task = "regression"
        output = "continuous_value"
        loss = "mse"
        metrics = ["mae", "rmse"]
    elif any(token in text for token in ["segment", "mask", "iou", "dice"]):
        task = "segmentation"
        domain = "CV"
        input_type = "image"
        output = "mask"
        model_type = "unet"
        loss = "cross_entropy"
        metrics = ["iou", "dice"]
        components = ["encoder", "decoder", "skip_connections"]
    elif any(token in text for token in ["tabular", "feature vector", "xgboost", "csv"]):
        domain = "tabular"
        input_type = "feature_vector"
        model_type = "mlp"
        components = ["dense_layers", "normalization", "prediction_head"]
    elif any(token in text for token in ["generation", "autoregressive", "next token", "language model"]):
        domain = "NLP"
        input_type = "text"
        task = "generation"
        output = "generated_text"
        model_type = "causal_lm"
        loss = "cross_entropy"
        metrics = ["perplexity"]
        components = ["embedding", "decoder", "lm_head"]

    if "unknown" in text:
        ambiguities.append("Parsed text explicitly contained uncertain descriptions.")
    if domain == "NLP" and "transformer" not in text and "bert" not in text and "attention" not in text:
        assumptions.append("NLP baseline may use a non-transformer architecture if the paper is not explicit.")

    return PaperAnalysis(
        task=EvidenceField(value=task, confidence=0.55, source_section="heuristic", assumed=True),
        domain=EvidenceField(value=domain, confidence=0.55, source_section="heuristic", assumed=True),
        input_data_type=EvidenceField(value=input_type, confidence=0.55, source_section="heuristic", assumed=True),
        output_format=EvidenceField(value=output, confidence=0.55, source_section="heuristic", assumed=True),
        model_type=EvidenceField(value=model_type, confidence=0.5, source_section="heuristic", assumed=True),
        components=EvidenceField(value=components, confidence=0.45, source_section="heuristic", assumed=True),
        loss=EvidenceField(value=loss, confidence=0.55, source_section="heuristic", assumed=True),
        metrics=EvidenceField(value=metrics, confidence=0.5, source_section="heuristic", assumed=True),
        training_details=TrainingDetails(),
        ambiguities=ambiguities,
        assumptions=assumptions,
    )
