from app.models.schemas import PaperAnalysis


def suggest_datasets(analysis: PaperAnalysis) -> list[str]:
    if analysis.domain == "NLP":
        if analysis.task == "classification":
            return ["ag_news", "dbpedia_14", "yelp_review_full"]
        return ["imdb"]
    if analysis.domain == "CV":
        if analysis.task == "classification":
            return ["cifar10", "mnist"]
    return []
