# Transformer Model Geometrical Analysis

## Overview

This project investigates the Transformer model, specifically focusing on Chat GPT-2 small, from a geometrical perspective. Our team dissected the model to inspect the algorithm and its parameters, studying the dimensionality of the embedding space. We observed how words, embedded in a 768-dimensional space, generally span a subspace due to the structure present in meaningful text. This analysis was conducted layer by layer, decoder by decoder, using both global and local methods such as intrinsic dimensionality estimation.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Contributors](#contributors)

## Introduction

The goal of this project is to gain a deeper understanding of the internal workings of the Chat GPT-2 small model through a geometrical lens. We analyzed the embedding space's dimensionality, studying how the dimension changes throughout the model’s layers and decoders. Our investigation also focused on the evolution of various metrics and the behavior of the model concerning the last word in a prompt, as it plays a crucial role in predicting the next word in the sequence.

## Installation

To get started with this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/adadiorio/Project-LCP-mod-B
    cd transformer-geometrical-analysis
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the analysis, execute:
-  Decoderwise_statistical_analysis: 
-  IDwise_statistical_analysis:
-  DimensionalityEvolution:

## Contributors

- **[Ginevra Beltrame](https://github.com/ginevrabeltrame)**
- **[Emanuele Coradin](https://github.com/EmanueleCoradin)**
- **[Ada Diorio](https://github.com/adadiorio)**
- **[Dario Liotta](https://github.com/darioliotta)**
