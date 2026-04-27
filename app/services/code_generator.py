from __future__ import annotations

from pathlib import Path

import yaml

from app.models.schemas import DecisionConfig, JobRecord
from app.services.project_writer import write_text


def _analysis_value(job: JobRecord, name: str, default: str) -> str:
    if not job.analysis:
        return default
    field = getattr(job.analysis, name, None)
    if field is None:
        return default
    value = getattr(field, "value", default)
    return str(value)


def _config_payload(job: JobRecord, decision_config: DecisionConfig) -> dict:
    domain = _analysis_value(job, "domain", "unknown").lower()
    task = _analysis_value(job, "task", "unknown").lower()
    input_data_type = _analysis_value(job, "input_data_type", "unknown").lower()
    components: list[str] = []
    if job.analysis:
        raw_components = getattr(job.analysis.components, "value", [])
        if isinstance(raw_components, list):
            components = [str(item) for item in raw_components]
    return {
        "paper_title": job.parsed_paper.title if job.parsed_paper else "",
        "paper_problem": job.parsed_paper.problem if job.parsed_paper else "",
        "domain": domain,
        "task": task,
        "input_data_type": input_data_type,
        "dataset": decision_config.dataset.model_dump(),
        "model": {
            **decision_config.model.model_dump(),
            "components": components,
        },
        "training": decision_config.training.model_dump(),
        "assumptions": decision_config.assumptions,
        "implementation_notes": job.plan.implementation_notes if job.plan else [],
    }


def _model_py() -> str:
    return '''"""
Paper2Project generated models.
The module supports multiple baseline families so the same project structure can
cover NLP, CV, tabular, and RL starter implementations.
"""

import torch
from torch import nn


class TextClassifier(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, num_classes: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        _, hidden = self.encoder(embedded)
        return self.classifier(hidden[-1])


class CNNClassifier(nn.Module):
    def __init__(self, num_channels: int, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


class SimpleUNet(nn.Module):
    def __init__(self, num_channels: int, num_classes: int):
        super().__init__()
        self.encoder1 = nn.Sequential(nn.Conv2d(num_channels, 32, 3, padding=1), nn.ReLU(), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU())
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU())
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decode1 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decode2 = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU())
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        enc1 = self.encoder1(inputs)
        enc2 = self.encoder2(self.pool1(enc1))
        bottleneck = self.bottleneck(self.pool2(enc2))
        dec1 = self.up1(bottleneck)
        dec1 = self.decode1(torch.cat([dec1, enc2], dim=1))
        dec2 = self.up2(dec1)
        dec2 = self.decode2(torch.cat([dec2, enc1], dim=1))
        return self.head(dec2)


class TabularMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class DQNNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class TextLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.encoder = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(inputs)
        outputs, _ = self.encoder(embedded)
        return self.head(outputs)


def build_model(config: dict, metadata: dict) -> nn.Module:
    domain = config["domain"]
    task = config["task"]
    if domain == "cv" and task == "segmentation":
        return SimpleUNet(metadata["num_channels"], metadata["num_outputs"])
    if domain == "cv" and task == "classification":
        return CNNClassifier(metadata["num_channels"], metadata["num_outputs"])
    if domain == "tabular":
        return TabularMLP(metadata["input_dim"], metadata["num_outputs"])
    if domain == "rl":
        return DQNNetwork(metadata["state_dim"], metadata["action_dim"])
    if task == "generation":
        return TextLanguageModel(metadata["vocab_size"], 128)
    return TextClassifier(metadata["vocab_size"], 128, metadata["num_outputs"])
'''


def _data_loader_py() -> str:
    return '''"""
Paper2Project generated dataset loader.
Supports Hugging Face text datasets, torchvision datasets, sklearn tabular data,
Gymnasium environments, and synthetic fallback generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class DatasetBundle:
    train_loader: Any | None = None
    eval_loader: Any | None = None
    metadata: dict[str, Any] | None = None
    environment: Any | None = None


def _simple_tokenizer(texts, max_length: int):
    rows = []
    for text in texts:
        tokens = [min(ord(ch), 255) for ch in text[:max_length]]
        tokens += [0] * (max_length - len(tokens))
        rows.append(tokens)
    return torch.tensor(rows, dtype=torch.long)


def _prepare_huggingface_dataset(config: dict) -> DatasetBundle:
    from datasets import load_dataset

    dataset_name = config["dataset"]["selected"]
    dataset = load_dataset(dataset_name)
    train_split = dataset["train"]
    eval_split = dataset["test"] if "test" in dataset else dataset["validation"] if "validation" in dataset else dataset["train"]

    text_key = "text"
    label_key = "label"
    sample_row = train_split[0]
    if text_key not in sample_row:
        text_key = next((key for key, value in sample_row.items() if isinstance(value, str)), list(sample_row.keys())[0])
    if label_key not in sample_row:
        label_key = next((key for key, value in sample_row.items() if isinstance(value, int)), list(sample_row.keys())[-1])

    def collate_fn(batch):
        texts = [item[text_key] for item in batch]
        labels = [int(item[label_key]) for item in batch]
        token_ids = _simple_tokenizer(texts, config["training"]["max_length"])
        if config["task"] == "generation":
            return token_ids[:, :-1], token_ids[:, 1:]
        return token_ids, torch.tensor(labels, dtype=torch.long)

    metadata = {"vocab_size": 256, "num_outputs": len(set(train_split[label_key])) if config["task"] != "generation" else 256}
    return DatasetBundle(
        train_loader=DataLoader(train_split, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate_fn),
        eval_loader=DataLoader(eval_split, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate_fn),
        metadata=metadata,
    )


def _prepare_torchvision_dataset(config: dict) -> DatasetBundle:
    from torchvision import datasets, transforms

    dataset_name = config["dataset"]["selected"]
    if config["task"] == "segmentation" and dataset_name == "OxfordIIITPet":
        image_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
        target_transform = transforms.Compose([transforms.Resize((128, 128)), transforms.PILToTensor()])
        train_dataset = datasets.OxfordIIITPet(
            root="./data",
            split="trainval",
            target_types="segmentation",
            download=True,
            transform=image_transform,
            target_transform=target_transform,
        )
        eval_dataset = datasets.OxfordIIITPet(
            root="./data",
            split="test",
            target_types="segmentation",
            download=True,
            transform=image_transform,
            target_transform=target_transform,
        )

        def collate_fn(batch):
            images = torch.stack([item[0] for item in batch])
            masks = torch.stack([item[1].squeeze(0).long() for item in batch])
            return images, masks

        metadata = {"num_channels": 3, "num_outputs": 3}
        return DatasetBundle(
            train_loader=DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, collate_fn=collate_fn),
            eval_loader=DataLoader(eval_dataset, batch_size=config["training"]["batch_size"], shuffle=False, collate_fn=collate_fn),
            metadata=metadata,
        )

    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    dataset_cls = getattr(datasets, dataset_name)
    train_dataset = dataset_cls(root="./data", train=True, download=True, transform=transform)
    eval_dataset = dataset_cls(root="./data", train=False, download=True, transform=transform)
    sample, _ = train_dataset[0]
    metadata = {
        "num_channels": int(sample.shape[0]),
        "num_outputs": len(train_dataset.classes) if hasattr(train_dataset, "classes") else 10,
    }
    return DatasetBundle(
        train_loader=DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True),
        eval_loader=DataLoader(eval_dataset, batch_size=config["training"]["batch_size"], shuffle=False),
        metadata=metadata,
    )


def _prepare_sklearn_dataset(config: dict) -> DatasetBundle:
    dataset_name = config["dataset"]["selected"]
    loaders = {
        "breast_cancer": load_breast_cancer,
        "wine": load_wine,
        "iris": load_iris,
        "diabetes": load_diabetes,
        "california_housing": fetch_california_housing,
    }
    data = loaders[dataset_name]()
    features = np.asarray(data.data, dtype=np.float32)
    targets = np.asarray(data.target)
    if config["task"] == "regression":
        targets = targets.astype(np.float32).reshape(-1, 1)
    else:
        targets = targets.astype(np.int64)
    scaler = StandardScaler()
    features = scaler.fit_transform(features).astype(np.float32)
    x_train, x_eval, y_train, y_eval = train_test_split(features, targets, test_size=0.2, random_state=config["training"]["seed"])
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_eval = torch.tensor(x_eval, dtype=torch.float32)
    if config["task"] == "regression":
        y_train = torch.tensor(y_train, dtype=torch.float32)
        y_eval = torch.tensor(y_eval, dtype=torch.float32)
        num_outputs = 1
    else:
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_eval = torch.tensor(y_eval, dtype=torch.long)
        num_outputs = int(len(np.unique(targets)))
    train_dataset = TensorDataset(x_train, y_train)
    eval_dataset = TensorDataset(x_eval, y_eval)
    metadata = {"input_dim": int(x_train.shape[1]), "num_outputs": num_outputs}
    return DatasetBundle(
        train_loader=DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True),
        eval_loader=DataLoader(eval_dataset, batch_size=config["training"]["batch_size"], shuffle=False),
        metadata=metadata,
    )


def _prepare_gymnasium_dataset(config: dict) -> DatasetBundle:
    import gymnasium as gym

    env_name = config["dataset"]["selected"]
    env = gym.make(env_name)
    metadata = {"state_dim": int(env.observation_space.shape[0]), "action_dim": int(env.action_space.n)}
    return DatasetBundle(environment=env, metadata=metadata)


def _prepare_synthetic_dataset(config: dict) -> DatasetBundle:
    task = config["task"]
    domain = config["domain"]
    if task == "segmentation":
        images = torch.randn(128, 3, 64, 64)
        masks = torch.randint(0, 2, (128, 64, 64))
        dataset = TensorDataset(images, masks)
        return DatasetBundle(
            train_loader=DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True),
            eval_loader=DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False),
            metadata={"num_channels": 3, "num_outputs": 2},
        )
    if domain == "tabular" or task == "regression":
        x = torch.randn(512, 16)
        if task == "regression":
            y = torch.sum(x[:, :4], dim=1, keepdim=True)
            metadata = {"input_dim": 16, "num_outputs": 1}
        else:
            y = (torch.sum(x[:, :4], dim=1) > 0).long()
            metadata = {"input_dim": 16, "num_outputs": 2}
        dataset = TensorDataset(x, y)
        return DatasetBundle(
            train_loader=DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True),
            eval_loader=DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False),
            metadata=metadata,
        )
    texts = [f"sample text {idx}" for idx in range(256)]
    labels = [idx % 2 for idx in range(256)]
    tokens = _simple_tokenizer(texts, config["training"]["max_length"])
    if task == "generation":
        dataset = TensorDataset(tokens[:, :-1], tokens[:, 1:])
        return DatasetBundle(
            train_loader=DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True),
            eval_loader=DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False),
            metadata={"vocab_size": 256, "num_outputs": 256},
        )
    dataset = TensorDataset(tokens, torch.tensor(labels, dtype=torch.long))
    return DatasetBundle(
        train_loader=DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True),
        eval_loader=DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=False),
        metadata={"vocab_size": 256, "num_outputs": 2},
    )


def build_data_bundle(config: dict) -> DatasetBundle:
    source = config["dataset"]["source"].lower()
    if source == "huggingface":
        return _prepare_huggingface_dataset(config)
    if source == "torchvision":
        return _prepare_torchvision_dataset(config)
    if source == "sklearn":
        return _prepare_sklearn_dataset(config)
    if source == "gymnasium":
        return _prepare_gymnasium_dataset(config)
    return _prepare_synthetic_dataset(config)
'''


def _train_py() -> str:
    return '''"""
Paper2Project generated training entrypoint.
Reads config.yaml, builds the right baseline, runs training, and logs metrics.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import yaml
from torch import nn
from torch.optim import Adam, AdamW
from torch.utils.tensorboard import SummaryWriter

from data_loader import build_data_bundle
from model import build_model


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_optimizer(config: dict, model: nn.Module):
    name = str(config["training"]["optimizer"]).lower()
    lr = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"].get("weight_decay", 0.0))
    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_loss(config: dict):
    loss_name = str(config["training"]["loss"]).lower()
    if loss_name in {"mse", "mse_loss"}:
        return nn.MSELoss()
    return nn.CrossEntropyLoss()


def evaluate_supervised(model, dataloader, device, task: str):
    model.eval()
    loss_values = []
    total = 0
    correct = 0
    criterion = nn.MSELoss() if task == "regression" else nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            if task == "generation":
                loss = nn.CrossEntropyLoss()(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
                loss_values.append(float(loss.item()))
            elif task == "segmentation":
                loss = nn.CrossEntropyLoss()(outputs, labels.long())
                loss_values.append(float(loss.item()))
            elif task == "regression":
                labels = labels.float()
                outputs = outputs.float()
                loss = criterion(outputs, labels)
                loss_values.append(float(loss.item()))
            else:
                loss = criterion(outputs, labels)
                loss_values.append(float(loss.item()))
                predictions = outputs.argmax(dim=-1)
                correct += int((predictions == labels).sum().item())
                total += int(labels.numel())
    metrics = {"loss": float(np.mean(loss_values)) if loss_values else 0.0}
    if task == "generation":
        metrics["perplexity"] = float(np.exp(min(metrics["loss"], 20.0)))
    elif task == "regression":
        metrics["mae_proxy"] = metrics["loss"]
    elif task == "segmentation":
        metrics["mask_loss"] = metrics["loss"]
    else:
        metrics["accuracy"] = (correct / total) if total else 0.0
    return metrics


def train_supervised(config: dict, device: str):
    bundle = build_data_bundle(config)
    model = build_model(config, bundle.metadata).to(device)
    optimizer = build_optimizer(config, model)
    criterion = build_loss(config)
    writer = SummaryWriter() if config["training"].get("use_tensorboard", True) else None
    wandb_run = None
    if config["training"].get("use_wandb", False):
        try:
            import wandb
            wandb_run = wandb.init(project="paper2project", config=config)
        except Exception:
            wandb_run = None

    for epoch in range(config["training"]["epochs"]):
        model.train()
        epoch_losses = []
        for inputs, labels in bundle.train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if config["task"] == "generation":
                loss = nn.CrossEntropyLoss()(outputs.reshape(-1, outputs.shape[-1]), labels.reshape(-1))
            elif config["task"] == "segmentation":
                loss = criterion(outputs, labels.long())
            elif config["task"] == "regression":
                labels = labels.float()
                outputs = outputs.float()
                loss = criterion(outputs, labels)
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        metrics = evaluate_supervised(model, bundle.eval_loader, device, config["task"])
        metrics["train_loss"] = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        print({"epoch": epoch + 1, "metrics": metrics})
        if writer:
            for key, value in metrics.items():
                writer.add_scalar(key, value, epoch + 1)
        if wandb_run:
            wandb_run.log({"epoch": epoch + 1, **metrics})

    if writer:
        writer.close()
    if wandb_run:
        wandb_run.finish()


def train_rl(config: dict, device: str):
    bundle = build_data_bundle(config)
    env = bundle.environment
    metadata = bundle.metadata
    model = build_model(config, metadata).to(device)
    target_model = build_model(config, metadata).to(device)
    target_model.load_state_dict(model.state_dict())
    optimizer = build_optimizer(config, model)
    replay_buffer = deque(maxlen=2000)
    gamma = 0.99
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.97
    batch_size = int(config["training"]["batch_size"])

    def select_action(state):
        nonlocal epsilon
        if random.random() < epsilon:
            return env.action_space.sample()
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            return int(model(state_tensor).argmax(dim=-1).item())

    for episode in range(config["training"]["epochs"]):
        state, _ = env.reset(seed=config["training"]["seed"] + episode)
        total_reward = 0.0
        done = False
        while not done:
            action = select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay_buffer.append((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward

            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states = torch.tensor(np.array([item[0] for item in batch]), dtype=torch.float32, device=device)
                actions = torch.tensor([item[1] for item in batch], dtype=torch.long, device=device)
                rewards = torch.tensor([item[2] for item in batch], dtype=torch.float32, device=device)
                next_states = torch.tensor(np.array([item[3] for item in batch]), dtype=torch.float32, device=device)
                dones = torch.tensor([item[4] for item in batch], dtype=torch.float32, device=device)
                current_q = model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                with torch.no_grad():
                    target_q = rewards + gamma * (1.0 - dones) * target_model(next_states).max(dim=1).values
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        target_model.load_state_dict(model.state_dict())
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print({"episode": episode + 1, "episode_return": total_reward, "epsilon": epsilon})


def main():
    config = load_config()
    set_seed(int(config["training"]["seed"]))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if config["domain"] == "rl" or config["task"] == "reinforcement_learning":
        train_rl(config, device)
    else:
        train_supervised(config, device)


if __name__ == "__main__":
    main()
'''


def _generated_readme(job: JobRecord) -> str:
    title = job.parsed_paper.title if job.parsed_paper else "Unknown paper"
    return f"""# Generated Project

This project was generated from:

- Paper: {title}

The generated implementation is a reproducible baseline built from the paper analysis,
pipeline plan, and approved decision config. It is designed to run, be editable, and
surface assumptions clearly.
"""


def _requirements_txt() -> str:
    return "\n".join(
        [
            "torch>=2.3.0",
            "torchvision>=0.18.0",
            "datasets>=2.20.0",
            "scikit-learn>=1.5.0",
            "gymnasium>=0.29.1",
            "PyYAML>=6.0.1",
            "tensorboard>=2.17.0",
            "wandb>=0.19.0",
        ]
    )


def build_generated_project(job: JobRecord, output_dir: Path, decision_config: DecisionConfig) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "model.py": _model_py(),
        "data_loader.py": _data_loader_py(),
        "train.py": _train_py(),
        "README.md": _generated_readme(job),
        "requirements.txt": _requirements_txt(),
    }
    written_files: list[str] = []
    for filename, content in files.items():
        path = output_dir / filename
        write_text(path, content)
        written_files.append(str(path))

    config_path = output_dir / "config.yaml"
    config_payload = _config_payload(job, decision_config)
    write_text(config_path, yaml.safe_dump(config_payload, sort_keys=False))
    written_files.append(str(config_path))
    return written_files
