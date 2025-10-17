# HPLC Anomaly Detection in Cloud Laboratories

This repository contains code and models for automated anomaly detection in High-Performance Liquid Chromatography (HPLC) experiments conducted in cloud laboratories, as described in the manuscript "Machine Learning anomaly detection of automated HPLC experiments in the Cloud Laboratory."

## Overview

This project implements a machine learning framework for detecting air bubble contamination in HPLC experiments - a common yet challenging issue that typically requires expert analytical chemists to identify. Using a 1D convolutional neural network, the system can automatically detect anomalies in HPLC experiments using pressure traces as input.

## Dataset

- **25,423 HPLC pressure traces** across 6 parquet files (`hplc_pre_tracks_batch_1-6_of_6.parquet`)
- **961 expert annotations** in `annotated_data.csv` (700 air bubble cases, 261 normal)
- **100 example traces** in `ExamplePreasureTraces_100_List[List[float]].pkl`

## Repository Structure

```bash
├── annotated_data.csv                              # Expert annotations (961 samples)
├── hplc_pre_tracks_batch_[1-6]_of_6.parquet        # Complete pressure trace dataset (25,423 traces)
├── ExamplePreasureTraces_100_List[List[float]].pkl # Example pressure traces
├── DeployedModel_epoch310-step19282.ckpt           # Trained model checkpoint
├── hparams.yaml                                    # Model hyperparameters
├── models.py                                       # Model definition
├── datamodule.py                                   # Data module for loading and preprocessing
├── Inference_Example.ipynb                         # Example notebook for inference
├── VisualizationEDA_example.ipynb                  # Exploratory data analysis
├── VisualizationEDA_example.html                   # EDA notebook (HTML export)
└── README.md                                       # This file

```

## Usage

See `Inference_Example.ipynb` for a complete example of loading the trained model and running inference on pressure traces. The model outputs probabilities where values close to 1.0 indicate air bubble contamination.
