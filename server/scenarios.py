"""
server/scenarios.py — Episode catalogue for the WhyDidItFail environment.

Each scenario key maps to a dict with at least:
    difficulty  : "easy" | "medium" | "hard"
    failure_mode: canonical label (matches SYSTEM_PROMPT label rules)
    description : human-readable task description shown to the agent
    logs        : training‐curve data revealed by inspect_logs
    config      : hyperparameter block revealed by inspect_config
    gradients   : gradient-norm stats revealed by inspect_gradients (or None)
    required_inspections: ordered list the environment enforces via Feedback
    correct_diagnosis: exact label string the grader checks against
"""

from typing import Any, Dict

SCENARIOS: Dict[str, Dict[str, Any]] = {
    # ─────────────────────────── EASY ────────────────────────────────────────
    "overfitting_easy": {
        "difficulty": "easy",
        "failure_mode": "overfitting",
        "description": (
            "A 5-layer CNN was trained for 30 epochs on CIFAR-10. "
            "After the first few epochs the training loss went very low, "
            "but validation performance stopped improving. Diagnose the failure."
        ),
        "correct_diagnosis": "overfitting",
        "required_inspections": ["logs", "config"],
        "logs": {
            "epochs": list(range(1, 31)),
            "train_loss": [round(2.3 - i * 0.07, 3) for i in range(30)],
            "val_loss":   [round(2.3 - i * 0.02 + max(0, (i - 8) * 0.06), 3) for i in range(30)],
            "train_acc":  [round(min(0.999, 0.10 + i * 0.03), 3) for i in range(30)],
            "val_acc":    [round(max(0.54, 0.10 + i * 0.015 - max(0, (i - 8) * 0.012)), 3) for i in range(30)],
        },
        "config": {
            "model": "CNN-5layer",
            "optimizer": "Adam",
            "lr": 0.001,
            "batch_size": 64,
            "epochs": 30,
            "dropout": 0.3,
            "weight_decay": 0.01,
            "dataset": "CIFAR-10",
        },
        "gradients": None,
    },

    "underfitting_easy": {
        "difficulty": "easy",
        "failure_mode": "underfitting",
        "description": (
            "A small linear model was trained on MNIST for 20 epochs. "
            "Both training and validation accuracy hover near random baseline. "
            "Diagnose the failure."
        ),
        "correct_diagnosis": "underfitting",
        "required_inspections": ["logs", "config"],
        "logs": {
            "epochs": list(range(1, 21)),
            "train_loss": [round(2.30 - i * 0.002, 3) for i in range(20)],
            "val_loss":   [round(2.29 - i * 0.002, 3) for i in range(20)],
            "train_acc":  [round(0.10 + i * 0.001, 3) for i in range(20)],
            "val_acc":    [round(0.10 + i * 0.001, 3) for i in range(20)],
        },
        "config": {
            "model": "Linear(784->10)",
            "optimizer": "Adam",
            "lr": 0.001,
            "batch_size": 256,
            "epochs": 20,
            "dropout": 0.0,
            "weight_decay": 0.0,
            "dataset": "MNIST",
        },
        "gradients": None,
    },

    # ─────────────────────────── MEDIUM ──────────────────────────────────────
    "exploding_gradients_medium": {
        "difficulty": "medium",
        "failure_mode": "exploding gradients",
        "description": (
            "An RNN was trained on a text corpus. Training seemed fine for the "
            "first 3 epochs but then loss became NaN. Diagnose the failure."
        ),
        "correct_diagnosis": "exploding gradients",
        "required_inspections": ["logs", "gradients"],
        "logs": {
            "epochs": list(range(1, 11)),
            "train_loss": [2.95, 2.71, 2.48, float("nan"), float("nan"),
                           float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
            "val_loss":   [2.97, 2.75, 2.52, float("nan"), float("nan"),
                           float("nan"), float("nan"), float("nan"), float("nan"), float("nan")],
            "train_acc":  [0.11, 0.14, 0.17, None, None, None, None, None, None, None],
            "val_acc":    [0.11, 0.13, 0.16, None, None, None, None, None, None, None],
        },
        "config": {
            "model": "RNN-2layer",
            "optimizer": "SGD",
            "lr": 0.1,
            "batch_size": 32,
            "epochs": 10,
            "clip_grad_norm": None,
            "dataset": "WikiText-2",
        },
        "gradients": {
            "epoch_3": {"layer1_norm": 12450.3, "layer2_norm": 98732.1, "output_norm": 321045.7},
            "note": "Gradient norms exploded starting epoch 3 → NaN in epoch 4",
        },
    },

    "missing_regularization_medium": {
        "difficulty": "medium",
        "failure_mode": "missing regularization",
        "description": (
            "A ResNet-18 was trained on CIFAR-100 for 50 epochs. "
            "Late in training val_loss starts rising steadily. Diagnose the failure."
        ),
        "correct_diagnosis": "missing regularization",
        "required_inspections": ["logs", "config"],
        "logs": {
            "epochs": list(range(1, 51)),
            "train_loss": [round(max(0.05, 4.6 - i * 0.09), 3) for i in range(50)],
            "val_loss":   [round(4.6 - i * 0.05 + max(0, (i - 20) * 0.04), 3) for i in range(50)],
            "train_acc":  [round(min(0.999, 0.001 + i * 0.02), 3) for i in range(50)],
            "val_acc":    [round(max(0.35, 0.001 + i * 0.015 - max(0, (i - 20) * 0.008)), 3) for i in range(50)],
        },
        "config": {
            "model": "ResNet-18",
            "optimizer": "Adam",
            "lr": 0.001,
            "batch_size": 128,
            "epochs": 50,
            "dropout": 0.0,
            "weight_decay": 0.0,
            "dataset": "CIFAR-100",
        },
        "gradients": None,
    },

    "lr_too_high_medium": {
        "difficulty": "medium",
        "failure_mode": "learning rate too high",
        "description": (
            "A feedforward network was trained on tabular data. Training loss "
            "oscillates wildly across epochs. Diagnose the failure."
        ),
        "correct_diagnosis": "learning rate too high",
        "required_inspections": ["logs", "config"],
        "logs": {
            "epochs": list(range(1, 21)),
            "train_loss": [2.31, 0.45, 3.12, 0.38, 2.89, 0.42, 3.05, 0.40,
                           2.95, 0.44, 3.10, 0.41, 2.87, 0.39, 3.01, 0.43,
                           2.93, 0.46, 3.08, 0.42],
            "val_loss":   [2.35, 0.51, 3.20, 0.48, 2.94, 0.50, 3.15, 0.47,
                           3.00, 0.52, 3.18, 0.49, 2.92, 0.47, 3.09, 0.51,
                           2.98, 0.53, 3.14, 0.50],
            "train_acc":  [0.11, 0.78, 0.08, 0.80, 0.09, 0.79, 0.08, 0.80,
                           0.09, 0.78, 0.08, 0.80, 0.09, 0.81, 0.08, 0.79,
                           0.09, 0.77, 0.08, 0.79],
            "val_acc":    [0.10, 0.75, 0.07, 0.77, 0.08, 0.76, 0.07, 0.77,
                           0.08, 0.75, 0.07, 0.76, 0.08, 0.78, 0.07, 0.76,
                           0.08, 0.74, 0.07, 0.76],
        },
        "config": {
            "model": "FeedForward-3layer",
            "optimizer": "SGD",
            "lr": 5.0,
            "batch_size": 256,
            "epochs": 20,
            "dropout": 0.0,
            "weight_decay": 0.0,
            "dataset": "tabular",
        },
        "gradients": None,
    },

    # ─────────────────────────── HARD ────────────────────────────────────────
    "vanishing_gradients_hard": {
        "difficulty": "hard",
        "failure_mode": "vanishing gradients",
        "description": (
            "A deep 12-layer MLP with sigmoid activations was trained on ImageNet. "
            "Training makes almost no progress. Diagnose the failure."
        ),
        "correct_diagnosis": "vanishing gradients",
        "required_inspections": ["logs", "config", "gradients"],
        "logs": {
            "epochs": list(range(1, 26)),
            "train_loss": [round(6.91 - i * 0.004, 4) for i in range(25)],
            "val_loss":   [round(6.91 - i * 0.003, 4) for i in range(25)],
            "train_acc":  [round(0.001 + i * 0.0003, 4) for i in range(25)],
            "val_acc":    [round(0.001 + i * 0.0002, 4) for i in range(25)],
        },
        "config": {
            "model": "MLP-12layer",
            "optimizer": "SGD",
            "lr": 0.01,
            "batch_size": 256,
            "epochs": 25,
            "activation": "sigmoid",
            "dropout": 0.0,
            "weight_decay": 0.0,
            "dataset": "ImageNet",
        },
        "gradients": {
            "layer_1_norm":  1.2e-1,
            "layer_3_norm":  4.5e-3,
            "layer_6_norm":  8.2e-5,
            "layer_9_norm":  1.1e-7,
            "layer_12_norm": 3.4e-9,
            "note": "Gradient norms decay exponentially towards input layers",
        },
    },

    "dying_relu_hard": {
        "difficulty": "hard",
        "failure_mode": "dying relu",
        "description": (
            "A 10-layer ResNet variant with ReLU activations was trained. "
            "Hidden layer gradient norms are exactly 0. Diagnose the failure."
        ),
        "correct_diagnosis": "dying relu",
        "required_inspections": ["logs", "config", "gradients"],
        "logs": {
            "epochs": list(range(1, 21)),
            "train_loss": [round(2.30 - i * 0.001, 3) for i in range(20)],
            "val_loss":   [round(2.30 - i * 0.001, 3) for i in range(20)],
            "train_acc":  [round(0.10 + i * 0.0005, 4) for i in range(20)],
            "val_acc":    [round(0.10 + i * 0.0005, 4) for i in range(20)],
        },
        "config": {
            "model": "ResNet-10-variant",
            "optimizer": "Adam",
            "lr": 0.1,
            "batch_size": 64,
            "activation": "relu",
            "epochs": 20,
            "dropout": 0.0,
            "weight_decay": 0.0,
            "dataset": "CIFAR-10",
        },
        "gradients": {
            "layer_1_norm":  0.0,
            "layer_3_norm":  0.0,
            "layer_5_norm":  0.0,
            "layer_7_norm":  0.0,
            "output_norm":   0.82,
            "note": "All hidden-layer gradient norms are exactly 0.0 — neurons are dead",
        },
    },

    "bad_weight_init_hard": {
        "difficulty": "hard",
        "failure_mode": "bad weight initialization",
        "description": (
            "A transformer encoder was trained on NLP data. "
            "Training loss is NaN from the very first epoch. Diagnose the failure."
        ),
        "correct_diagnosis": "bad weight initialization",
        "required_inspections": ["logs", "config", "gradients"],
        "logs": {
            "epochs": list(range(1, 11)),
            "train_loss": [float("nan")] * 10,
            "val_loss":   [float("nan")] * 10,
            "train_acc":  [None] * 10,
            "val_acc":    [None] * 10,
        },
        "config": {
            "model": "TransformerEncoder-6layer",
            "optimizer": "Adam",
            "lr": 0.0001,
            "batch_size": 32,
            "epochs": 10,
            "weight_init": "normal(mean=0, std=100)",
            "dropout": 0.1,
            "weight_decay": 0.01,
            "dataset": "BookCorpus",
        },
        "gradients": {
            "epoch_1_layer1_norm": 1_245_308.4,
            "epoch_1_layer3_norm": 8_921_045.2,
            "epoch_1_layer6_norm": 45_320_187.9,
            "note": "Gradient norms > 10000 from step 1 due to extreme weight std=100",
        },
    },

    "optimizer_misconfig_hard": {
        "difficulty": "hard",
        "failure_mode": "optimizer misconfiguration",
        "description": (
            "A VGG-16 was trained from scratch on CIFAR-10 using SGD. "
            "Both train and val loss stay high with no improvement after many epochs. "
            "Diagnose the failure."
        ),
        "correct_diagnosis": "optimizer misconfiguration",
        "required_inspections": ["logs", "config"],
        "logs": {
            "epochs": list(range(1, 31)),
            "train_loss": [round(2.30 - i * 0.001, 3) for i in range(30)],
            "val_loss":   [round(2.30 - i * 0.001, 3) for i in range(30)],
            "train_acc":  [round(0.10 + i * 0.0005, 4) for i in range(30)],
            "val_acc":    [round(0.10 + i * 0.0005, 4) for i in range(30)],
        },
        "config": {
            "model": "VGG-16",
            "optimizer": "SGD",
            "momentum": 0.0,
            "lr": 0.01,
            "batch_size": 64,
            "epochs": 30,
            "dropout": 0.5,
            "weight_decay": 0.0005,
            "dataset": "CIFAR-10",
        },
        "gradients": None,
    },
}
