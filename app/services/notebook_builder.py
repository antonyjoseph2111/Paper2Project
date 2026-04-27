from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from app.models.schemas import DecisionConfig, JobRecord


def build_colab_notebook(job: JobRecord, output_dir: Path, decision_config: DecisionConfig) -> str:
    notebook = nbf.v4.new_notebook()
    paper_title = job.parsed_paper.title if job.parsed_paper else "Unknown paper"
    dataset_name = decision_config.dataset.selected
    training = decision_config.training.model_dump()

    notebook.cells = [
        nbf.v4.new_markdown_cell(f"# Paper2Project Notebook\n\nGenerated from **{paper_title}**."),
        nbf.v4.new_code_cell("!pip install datasets torch pyyaml"),
        nbf.v4.new_code_cell(
            "import random\nimport numpy as np\nimport torch\n\n"
            f"SEED = {training['seed']}\n"
            "random.seed(SEED)\nnp.random.seed(SEED)\ntorch.manual_seed(SEED)\n"
        ),
        nbf.v4.new_code_cell(
            f"CONFIG = {training}\nDATASET_NAME = '{dataset_name}'\nprint(CONFIG)\nprint(DATASET_NAME)"
        ),
        nbf.v4.new_markdown_cell("## Dataset download"),
        nbf.v4.new_code_cell(
            "from datasets import load_dataset\n"
            "dataset = load_dataset(DATASET_NAME)\n"
            "dataset"
        ),
        nbf.v4.new_markdown_cell("## Model definition"),
        nbf.v4.new_code_cell(
            "import torch\nfrom torch import nn\n\n"
            "class PaperModel(nn.Module):\n"
            "    def __init__(self, vocab_size=30522, embed_dim=128, num_classes=4):\n"
            "        super().__init__()\n"
            "        self.embedding = nn.Embedding(vocab_size, embed_dim)\n"
            "        self.encoder = nn.GRU(embed_dim, embed_dim, batch_first=True)\n"
            "        self.classifier = nn.Linear(embed_dim, num_classes)\n\n"
            "    def forward(self, input_ids):\n"
            "        embedded = self.embedding(input_ids)\n"
            "        _, hidden = self.encoder(embedded)\n"
            "        return self.classifier(hidden[-1])\n\n"
            "model = PaperModel()\nmodel"
        ),
        nbf.v4.new_markdown_cell("## Training loop"),
        nbf.v4.new_code_cell(
            "def simple_tokenizer(texts, max_length=128):\n"
            "    rows = []\n"
            "    for text in texts:\n"
            "        tokens = [min(ord(ch), 255) for ch in text[:max_length]]\n"
            "        tokens += [0] * (max_length - len(tokens))\n"
            "        rows.append(tokens)\n"
            "    return torch.tensor(rows, dtype=torch.long)\n\n"
            "print('Tokenizer ready')"
        ),
        nbf.v4.new_markdown_cell("## Evaluation"),
        nbf.v4.new_code_cell("print('Add evaluation logic here or import from generated project files.')"),
        nbf.v4.new_markdown_cell("## Results visualization"),
        nbf.v4.new_code_cell("print({'status': 'baseline notebook generated'})"),
    ]

    notebook_path = output_dir / "paper2project_notebook.ipynb"
    with notebook_path.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    return str(notebook_path)
