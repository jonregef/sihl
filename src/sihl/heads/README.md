# Heads

Heads process features from backbones or necks to solve ML tasks.
Each head is associated to a task, and is responsible for outputting tensors of expected shapes at inference.
During training and validation, the head computes the task's loss and metrics.
They structurally subtype the [`Head`](./__init__.py) protocol.
