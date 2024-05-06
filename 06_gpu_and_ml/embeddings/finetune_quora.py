import json
import os
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import pandas as pd
from helpers import format_dataset, score_prediction
from modal import Image, Stub, Volume, gpu
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# GPU Configuration
gpu_config = gpu.A100()

# Finetuning Configuration ( Arrays are configurable parameters )
MODELS = [
    "sentence-transformers/all-mpnet-base-v2",
    "all-MiniLM-L6-v2",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",  # needs A100
]
SCHEDULER = [
    "warmuplinear",
]
DATASET_SIZE = [100, 1000, 10_000]
WARMUP_STEPS = [1500]
DENSE_OUT_FEATURES = [64, 512]
BATCH_SIZE = [32]
MODEL_SAVE_PATH = "/output"
MIN_LEARNING_RATE = 1e-5
MAX_LEARNING_RATE = 1e-3
MAX_EPOCHS = 8

# DATASET CONFIG
VOLUME = Volume.from_name("finetune-quora", create_if_missing=True)
VOLUME_ROOT = Path("/data")
DATASET_NAME = "567-labs/cleaned-quora-dataset-train-test-split"
DATASET_PATH = VOLUME_ROOT / "quora-dataset"
JOURNAL_PATH = VOLUME_ROOT / "log"

TEST_SET_SIZE = 10000

# Eval Configuration
METRICS = {
    "accuracy": accuracy_score,  # This is the number of correct predictions by the model ( TP + TN )/ (# of samples)
    "precision": precision_score,  # This measures the number of positive class predicitons (TP) / (TP + FP)
    "recall": recall_score,  # This measure the number of negative class predictions (TP) / ( TP + FN )
    "AUC": roc_auc_score,
}

stub = Stub("finetune-quora")


def download_model():
    from sentence_transformers import SentenceTransformer

    for model in MODELS:
        SentenceTransformer(model)


image = (
    Image.debian_slim()
    .pip_install(
        "sentence-transformers", "torch", "datasets", "optuna", "pandas"
    )
    .run_function(download_model)
    .apt_install("procps")
)


@stub.function(image=image, volumes={VOLUME_ROOT: VOLUME})
def download_dataset():
    from datasets import load_dataset

    if os.path.exists(DATASET_PATH):
        print("Dataset Exists")
        return

    dataset = load_dataset(DATASET_NAME)

    dataset.save_to_disk(DATASET_PATH.as_posix())
    VOLUME.commit()


@dataclass
class ModelConfig:
    model_name: str
    dataset_size: int
    dense_out_features: int = 64
    freeze_embedding_model: bool = True
    learning_rate: float = 1e-4
    batch_size: int = 32
    warmup_steps: int = 500
    scheduler: str = "warmuplinear"
    num_epochs: int = 8
    train: bool = True


@stub.function(
    image=image,
    gpu=gpu_config,
    volumes={VOLUME_ROOT: VOLUME},
    # concurrency_limit=1,
    timeout=2 * 60 * 60,
)
def objective(
    config: ModelConfig,
):
    import os
    import shutil

    import torch.nn as nn
    from datasets import load_from_disk
    from sentence_transformers import (
        SentenceTransformer,
        evaluation,
        losses,
        models,
    )
    from torch.utils.data import DataLoader

    model_name = config.model_name
    dataset_size = config.dataset_size
    dense_out_features = config.dense_out_features
    learning_rate = config.learning_rate
    scheduler = config.scheduler
    warmup_steps = config.warmup_steps
    freeze_embedding_model = config.freeze_embedding_model
    batch_size = config.batch_size
    num_epochs = config.num_epochs

    print(f"Training model {model_name} {config}")
    model_slug = model_name.replace("/", "-")
    journal_file_name = (
        JOURNAL_PATH
        / f"{model_slug}_{dataset_size}_{dense_out_features}_{freeze_embedding_model}_{batch_size}_{num_epochs}.json"
    )
    print(journal_file_name)

    if journal_file_name.exists():
        with open(journal_file_name, "r") as f:
            return json.load(f)
    # with open(journal_file_name, "w") as f:
    #     return json.load(f)

    # Load the model
    embedding_model = SentenceTransformer(model_name)

    # Delete the directory if it exists
    if os.path.exists(MODEL_SAVE_PATH):
        shutil.rmtree(MODEL_SAVE_PATH)

    # Model configuration
    dim_emb = embedding_model.get_sentence_embedding_dimension()

    # Freeze the embedding model
    if freeze_embedding_model:
        for param in embedding_model._first_module().auto_model.parameters():
            param.requires_grad = False

    # Define the model architecture with additional dense layer
    dense_model = models.Dense(
        in_features=dim_emb,
        out_features=dense_out_features,
        activation_function=nn.Tanh(),
    )
    pooling_model = models.Pooling(dim_emb)

    # Initialize the model
    model = SentenceTransformer(
        modules=[embedding_model, pooling_model, dense_model], device="cuda"
    )

    # Load the dataset
    dataset = load_from_disk(DATASET_PATH.as_posix())
    train_dataset = dataset["train"].select(range(dataset_size))
    test_dataset = dataset["test"].select(range(TEST_SET_SIZE))

    if config.train:
        # Format the dataset
        train_examples, test_examples = (
            format_dataset(train_dataset),
            format_dataset(test_dataset),
        )

        # Create dataloaders and evaluator
        train_dataloader = DataLoader(
            train_examples, shuffle=True, batch_size=batch_size
        )
        evaluator = (
            evaluation.BinaryClassificationEvaluator.from_input_examples(
                test_examples, batch_size=batch_size
            )
        )
        train_loss = losses.OnlineContrastiveLoss(model)

        # Train the model
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            evaluator=evaluator,
            warmup_steps=warmup_steps,
            scheduler=scheduler,
            optimizer_params={"lr": learning_rate},
            save_best_model=True,
            epochs=num_epochs,
            output_path=MODEL_SAVE_PATH,
        )

        # Reload the best model
        model = SentenceTransformer(MODEL_SAVE_PATH)
    else:
        print("Skipping training")

    # Score and evaluate the model
    predictions, test_labels = score_prediction(
        model, train_dataset, test_dataset
    )
    eval_results = {
        f"metric_{metric}": round(function(test_labels, predictions), 4)
        for metric, function in METRICS.items()
    }
    eval_results["model_name"] = model_name
    eval_results["dataset_size"] = dataset_size
    eval_results["dense_out_features"] = dense_out_features
    eval_results["learning_rate"] = learning_rate
    eval_results["scheduler"] = scheduler
    eval_results["warmup_steps"] = warmup_steps
    eval_results["freeze_embedding_model"] = freeze_embedding_model
    eval_results["batch_size"] = batch_size
    eval_results["num_epochs"] = num_epochs
    eval_results["train"] = config.train

    print(json.dumps(eval_results, indent=2))

    journal_file_name.parent.mkdir(parents=True, exist_ok=True)
    with open(journal_file_name, "w") as f:
        json.dump(eval_results, f)
        VOLUME.commit()

    return eval_results


def generate_configs():
    for model in MODELS:
        yield ModelConfig(model_name=model, dataset_size=1_000, train=False)

    for (
        model,
        sample_size,
        freeze_embedding_model,
        dense_out_features,
    ) in product(MODELS, DATASET_SIZE, [True, False], DENSE_OUT_FEATURES):
        yield ModelConfig(
            model_name=model,
            dataset_size=sample_size,
            freeze_embedding_model=freeze_embedding_model,
            dense_out_features=dense_out_features,
        )


@stub.local_entrypoint()
def main():
    results = list(objective.map(generate_configs(), order_outputs=True))

    df = pd.DataFrame(results).sort_values("metric_accuracy", ascending=False)
    df.to_csv("./embedding_finetune.csv", index=False)
