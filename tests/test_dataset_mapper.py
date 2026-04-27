from app.models.schemas import EvidenceField, PaperAnalysis
from app.services.dataset_mapper import suggest_datasets


def test_dataset_mapper_handles_rl() -> None:
    analysis = PaperAnalysis(
        task=EvidenceField(value="reinforcement_learning"),
        domain=EvidenceField(value="RL"),
        input_data_type=EvidenceField(value="state_vector"),
        output_format=EvidenceField(value="action"),
        model_type=EvidenceField(value="dqn"),
        components=EvidenceField(value=["q_network"]),
        loss=EvidenceField(value="temporal_difference"),
        metrics=EvidenceField(value=["episode_return"]),
    )
    result = suggest_datasets(analysis)
    assert result["source"] == "gymnasium"
    assert "CartPole-v1" in result["candidates"]
