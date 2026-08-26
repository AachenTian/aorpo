# Uncertainty-Aware AORPO — FACMAC Evaluation

Additional evaluation of the uncertainty-aware AORPO method on the JaxMARL FACMAC environment.

This branch accompanies the thesis:

**Model-based Multi-agent Reinforcement Learning via Uncertainty Quantification**

The primary implementation and Simple Spread experiments are available on the `main` branch.

## Environment

This branch uses:

- `MPE_simple_facmac_3a_v1`
- 3 agents
- FACMAC pursuit task

## Installation

    pip install -r requirements.txt
    pip install --no-deps "jaxmarl==0.1.0"

For GPU execution, install the appropriate GPU-enabled JAX wheel for your CUDA environment.

## Training

Run from the repository root:

    python train.py

The default configuration is located at:

    aorpo/configs/train.yaml

## Weights & Biases

Experiment tracking with Weights & Biases is supported.

The project and entity can be configured through environment variables:

    WANDB_PROJECT=AORPO-UQ-FACMAC WANDB_ENTITY=<your-entity> python train.py

For offline logging:

    WANDB_MODE=offline python train.py

## Main Implementation

The primary uncertainty-aware AORPO implementation is available on the `main` branch.
