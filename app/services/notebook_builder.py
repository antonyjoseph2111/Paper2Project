from __future__ import annotations

from pathlib import Path

import nbformat as nbf

from app.models.schemas import DecisionConfig, JobRecord


def build_colab_notebook(job: JobRecord, output_dir: Path, decision_config: DecisionConfig) -> str:
    paper_title = job.parsed_paper.title if job.parsed_paper else "Unknown paper"
    notebook = nbf.v4.new_notebook(
        metadata={
            "colab": {"name": "paper2project_notebook.ipynb", "provenance": [], "gpuType": "T4"},
            "accelerator": "GPU",
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        }
    )

    def safe_read(filename: str) -> str:
        path = output_dir / filename
        if not path.exists():
            return f"# Missing generated artifact: {filename}\n"
        return path.read_text(encoding="utf-8")

    model_py = safe_read("model.py")
    data_loader_py = safe_read("data_loader.py")
    train_py = safe_read("train.py")
    config_yaml = safe_read("config.yaml")

    notebook.cells = [
        nbf.v4.new_markdown_cell(
            f"# Paper2Project Notebook\n\nGenerated from **{paper_title}**.\n\nThis notebook recreates the generated project files and runs the training pipeline in Colab."
        ),
        nbf.v4.new_markdown_cell(
            "## Runtime recommendation\n\nUse **Runtime -> Change runtime type -> GPU** for faster training when the generated baseline supports it."
        ),
        nbf.v4.new_code_cell("!pip install -q torch torchvision datasets scikit-learn gymnasium pyyaml tensorboard wandb"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "project_dir = Path('/content/paper2project_generated')\n"
            "project_dir.mkdir(parents=True, exist_ok=True)\n"
            "project_dir"
        ),
        nbf.v4.new_markdown_cell("## Recreate generated files"),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            f"Path(project_dir / 'model.py').write_text({model_py!r}, encoding='utf-8')\n"
            "print('model.py written')"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            f"Path(project_dir / 'data_loader.py').write_text({data_loader_py!r}, encoding='utf-8')\n"
            "print('data_loader.py written')"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            f"Path(project_dir / 'train.py').write_text({train_py!r}, encoding='utf-8')\n"
            "print('train.py written')"
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            f"Path(project_dir / 'config.yaml').write_text({config_yaml!r}, encoding='utf-8')\n"
            "print('config.yaml written')"
        ),
        nbf.v4.new_code_cell(
            "import os\n"
            "os.chdir(project_dir)\n"
            "print('Working directory:', os.getcwd())\n"
            "print(sorted(path.name for path in Path('.').iterdir()))"
        ),
        nbf.v4.new_markdown_cell("## Inspect config"),
        nbf.v4.new_code_cell("print(Path('config.yaml').read_text(encoding='utf-8'))"),
        nbf.v4.new_markdown_cell("## Run training"),
        nbf.v4.new_code_cell("!python train.py"),
    ]

    notebook_path = output_dir / "paper2project_notebook.ipynb"
    with notebook_path.open("w", encoding="utf-8") as handle:
        nbf.write(notebook, handle)
    return str(notebook_path)
