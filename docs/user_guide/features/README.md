# Features

Speculators provides a comprehensive set of features for building, training, and deploying speculative decoding models. This section covers the core capabilities of the library.

## Overview

The features in Speculators span the entire workflow of speculative decoding model development:

- **Data Preparation** - Transform raw conversational datasets into training-ready format
- **Hidden States Generation** - Extract hidden states from verifier models for training
- **Model Training** - Train speculator models with flexible online or offline approaches
- **Response Regeneration** - Generate diverse training data with response variations
- **Model Conversion** - Convert third-party models to the Speculators format

## Available Features

### [Response Regeneration](response_regeneration.md)

Generate alternative responses to training data using the target model to create more diverse training datasets. This technique helps improve the robustness of speculator models by exposing them to a wider variety of language patterns.

### [Prepare Data](prepare_data.md)

Preprocess conversational datasets by applying chat templates, tokenization, and creating loss masks. This is the first step in the training pipeline that converts raw text into a format ready for speculator training.

### [Offline Hidden States Generation](offline_hidden_states.md)

Extract and save hidden states from the verifier model using vLLM for offline training. This approach pre-generates all training data, enabling faster training iterations and reduced GPU memory requirements during training.

### [Training](training.md)

Train speculator models using PyTorch with support for distributed training via FSDP. Choose between online training (generating hidden states on-demand) or offline training (using pre-generated hidden states).

### [Conversion](conversion.md)

Convert models from third-party formats (e.g., EAGLE repositories) into the standardized Speculators format. This allows you to use existing pre-trained models with vLLM and other Speculators-compatible tools.
