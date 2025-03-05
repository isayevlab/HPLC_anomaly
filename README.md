# HPLC Anomaly Detection in Cloud Laboratories
This repository contains code and models for automated anomaly detection
in High-Performance Liquid Chromatography (HPLC) experiments conducted in cloud laboratories,
as described in the manuscript "Machine Learning anomaly detection of automated HPLC experiments in the Cloud Laboratory."

## Overview

This project implements a machine learning framework for detecting air bubble contamination
in HPLC experiments - a common yet challenging issue that typically requires expert analytical
chemists to identify. Using a 1D convolutional neural network, the system can automatically detect anomalies in HPLC experiments using pressure traces as input.

## Repository Structure

```bash
├── datamodule.py                                   # Data module for loading and preprocessing the data
├── DeployedModel_epoch310-step19282.ckpt           # Trained model
├── ExamplePreasureTraces_100_List[List[float]].pkl # Example pressure traces
├── hparams.yaml                                    # Hyperparameters
├── Inference_Example.ipynb                         # Example notebook for inference
├── models.py                                       # Model definition
└── README.md                                       # This file
```