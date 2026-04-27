from __future__ import annotations

from typing import TypedDict

from app.models.schemas import PaperAnalysis


class DatasetSuggestion(TypedDict):
    source: str
    candidates: list[str]
    notes: str
    synthetic_fallback: str


def suggest_datasets(analysis: PaperAnalysis) -> DatasetSuggestion:
    task = str(analysis.task.value).lower()
    domain = str(analysis.domain.value).lower()

    if domain == "nlp":
        if task == "classification":
            return {
                "source": "huggingface",
                "candidates": ["ag_news", "dbpedia_14", "imdb"],
                "notes": "Fast text-classification datasets that run well in Colab.",
                "synthetic_fallback": "synthetic_text_classification",
            }
        if task == "generation":
            return {
                "source": "huggingface",
                "candidates": ["wikitext", "tiny_shakespeare"],
                "notes": "Language-model-friendly corpora for generation baselines.",
                "synthetic_fallback": "synthetic_language_modeling",
            }
    if domain == "cv":
        if task == "classification":
            return {
                "source": "torchvision",
                "candidates": ["CIFAR10", "FashionMNIST", "MNIST"],
                "notes": "Common vision classification datasets with simple loaders.",
                "synthetic_fallback": "synthetic_image_classification",
            }
        if task == "segmentation":
            return {
                "source": "torchvision",
                "candidates": ["OxfordIIITPet"],
                "notes": "Segmentation-friendly torchvision dataset with masks.",
                "synthetic_fallback": "synthetic_segmentation",
            }
    if domain == "tabular":
        if task == "regression":
            return {
                "source": "sklearn",
                "candidates": ["california_housing", "diabetes"],
                "notes": "Built-in tabular regression datasets for reliable local and Colab runs.",
                "synthetic_fallback": "synthetic_tabular_regression",
            }
        return {
            "source": "sklearn",
            "candidates": ["breast_cancer", "wine", "iris"],
            "notes": "Built-in tabular classification datasets for fast reproducible baselines.",
            "synthetic_fallback": "synthetic_tabular_classification",
        }
    if domain == "rl":
        return {
            "source": "gymnasium",
            "candidates": ["CartPole-v1", "MountainCar-v0"],
            "notes": "Small classic control tasks suitable for baseline RL experiments.",
            "synthetic_fallback": "CartPole-v1",
        }
    return {
        "source": "synthetic",
        "candidates": ["synthetic_placeholder"],
        "notes": "Unknown domain; use synthetic data until the user selects a better dataset.",
        "synthetic_fallback": "synthetic_placeholder",
    }
