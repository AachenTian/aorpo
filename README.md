# AORPO JAX Baseline

JAX reimplementation of the original AORPO baseline used for comparison in the thesis:

**Model-based Multi-agent Reinforcement Learning via Uncertainty Quantification**

This branch provides the baseline implementation used in the thesis experiments.
The uncertainty-aware extension is available on the `main` branch.

## Environment

The implementation uses the JaxMARL Simple Spread environment:

- `MPE_simple_spread_v3`
- 3 agents
- 3 landmarks
- cooperative navigation

## Installation

    pip install -r requirements.txt
    pip install --no-deps "jaxmarl==0.1.0"

For GPU execution, install the appropriate GPU-enabled JAX wheel for your CUDA environment.

## Training

Run training from the repository root:

    python train.py

The default configuration is located at:

    aorpo/configs/train.yaml

## Weights & Biases

Experiment tracking with Weights & Biases is supported.

The project and entity can be configured through environment variables:

    WANDB_PROJECT=AORPO-JAX WANDB_ENTITY=<your-entity> python train.py

For offline logging:

    WANDB_MODE=offline python train.py

## Main Implementation

The uncertainty-aware AORPO implementation developed in the thesis is available on the `main` branch.
