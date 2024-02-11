<p align="center">
  <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" width="100" />
</p>
<p align="center">
    <h1 align="center">HSF_TRAIN</h1>
</p>
<p align="center">
    <em>Empower MRI Insights: Pristine Precision, Effortless Integration</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/clementpoiret/hsf_train?style=flat&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/clementpoiret/hsf_train?style=flat&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/clementpoiret/hsf_train?style=flat&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/clementpoiret/hsf_train?style=flat&color=0080ff" alt="repo-language-count">
<p>
<p align="center">
		<em>Developed with the software and tools below.</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/YAML-CB171E.svg?style=flat&logo=YAML&logoColor=white" alt="YAML">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=flat&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/ONNX-005CED.svg?style=flat&logo=ONNX&logoColor=white" alt="ONNX">
</p>
<hr>

##  Quick Links

> - [ Overview](#-overview)
> - [ Features](#-features)
> - [ Repository Structure](#-repository-structure)
> - [ Modules](#-modules)
> - [ Getting Started](#-getting-started)
>   - [ Installation](#-installation)
>   - [Running hsf_train](#-running-hsf_train)
> - [ Changelog](#-changelog)
> - [ Contributing](#-contributing)
> - [ License](#-license)
> - [ Citing HSF](#-citing-hsf)

---

##  Overview

The hsf_train project is a sophisticated, configurable framework for fine-tuning, training, and exporting deep learning models tailored to MRI data segmentation, specifically focusing on hippocampal regions. Utilizing cutting-edge techniques, including custom segmentation models, data augmentation, and loss functions adapted from nnUNet, it strives for precision in medical imaging tasks. Key features include integration with SparseML for optimization—pruning and quantization—leading to deployment-ready models via ONNX export, and comprehensive experiment tracking with Weights & Biases. This project seamlessly combines flexible model architecture configurations, dataset management, and advanced training environments, offering an end-to-end solution for enhancing neural network performance and efficiency in medical image analysis.

---

##  Features

|    |   Feature         | Description |
|----|-------------------|---------------------------------------------------------------|
| ⚙️  | **Architecture**  | The project follows a modular design pattern and leverages the PyTorch Lightning framework. It includes configurable models, data loaders, and training configurations. The codebase supports fine-tuning of segmentation models, model export in ONNX format, and integration with SparseML for model optimization. |
| 🔩 | **Code Quality**  | The codebase exhibits good code quality and follows Python coding conventions. The code is well-structured and organized into modules and classes. It uses meaningful variable and function names, and includes type hints to improve readability and maintainability. The use of external libraries and dependencies is appropriate and follows best practices. |
| 📄 | **Documentation** | The project has well-documented code. The codebase includes inline comments explaining various functions and modules. Additionally, it provides configuration files with detailed descriptions of their purpose and available options. However, there could be room for improvement in terms of providing more comprehensive documentation and usage examples. |
| 🔌 | **Integrations**  | The codebase integrates with various external libraries and tools such as wandb (Weights and Biases), torchio, and SparseML. These integrations enhance the functionality of the project, enabling efficient logging, visualization, data augmentation, and model optimization. |
| 🧩 | **Modularity**    | The codebase demonstrates modularity and reusability. It separates concerns into different files and modules, allowing for easy extension and customization. The configuration files provide a flexible way to adjust various settings, making the codebase adaptable to different use cases and datasets. |
| 🧪 | **Testing**       | The project does not explicitly mention testing frameworks or tools. However, given its modular structure and code quality, it would be feasible to incorporate unit tests using frameworks like pytest or unittest to ensure code correctness and prevent regressions. |
| ⚡️  | **Performance**   | The performance of the codebase depends on the specific models and datasets used. The use of libraries like Lightning and SparseML suggests a focus on efficient deep learning training and optimization. However, a detailed evaluation of efficiency, speed, and resource usage would require benchmarking and profiling specific use cases. |
| 🛡️ | **Security**      | The project does not explicitly mention security measures. However, following Python security best practices, such as using secure dependencies and handling data securely, can help ensure data protection and access control. Regular dependency updates and code reviews can also mitigate security vulnerabilities. |
| 📦 | **Dependencies**  | The project relies on external libraries and dependencies such as rich, text, pymia, wandb, torch, xxhash, python, onnxruntime, sparseml, lightning, torchio, python-dotenv, yaml, and py. The 'requirements.txt' file specifies these dependencies, ensuring easy setup and reproducibility. |


---

##  Repository Structure

```sh
└── hsf_train/
    ├── LICENSE
    ├── conf
    │   ├── config.yaml
    │   ├── datasets
    │   │   ├── all.yaml
    │   │   └── custom_dataset.yaml
    │   ├── finetuning
    │   │   └── default.yaml
    │   ├── lightning
    │   │   └── default.yaml
    │   ├── logger
    │   │   └── wandb.yaml
    │   └── models
    │       └── default.yaml
    ├── finetune.py
    ├── hsftrain
    │   ├── callbacks.py
    │   ├── data
    │   │   ├── __init__.py
    │   │   ├── dataloader.py
    │   │   └── loader.py
    │   ├── exporter.py
    │   ├── models
    │   │   ├── __init__.py
    │   │   ├── blocks.py
    │   │   ├── helpers.py
    │   │   ├── layers.py
    │   │   ├── losses.py
    │   │   ├── models.py
    │   │   ├── optimizer.py
    │   │   ├── scheduler.py
    │   │   └── types.py
    │   └── utils.py
    ├── main.py
    ├── requirements.txt
    ├── scripts
    │   └── ckpt_to_onnx.py
    └── sparseml
        ├── finetuning.yaml
        └── scratch.yaml
```

---

##  Modules

<details closed><summary>.</summary>

| File                                                                                        | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---                                                                                         | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| [finetune.py](https://github.com/clementpoiret/hsf_train/blob/master/finetune.py)           | This script, `finetune.py`, fine-tunes a pre-trained segmentation model for MRI images with custom configurations, trains it with specific data augmentation, and logs performance metrics using Wandb. It supports exporting the fine-tuned model to ONNX format, with optional conversion to DeepSparse format if SparseML is utilized. Critical features include data preparation, model training, checkpointing, and exporting, aligning with the repository's focus on flexible, efficient deep learning workflows in medical imaging.                                                                                                                                          |
| [main.py](https://github.com/clementpoiret/hsf_train/blob/master/main.py)                   | The `main.py` script serves as the central entry point for a neural network training pipeline focused on MRI data segmentation within the larger `hsf_train` repository. It integrates data preprocessing, augmentation, and postprocessing workflows, utilizes a custom SegmentationModel with FocalTversky loss, and supports experiment tracking via Weights & Biases. Additionally, it includes optional SparseML integration for model pruning and quantization, leading to ONNX export functionalities for model deployment. This script emphasizes automated, configurable experimentation and model optimization for enhanced deployment-ready neural network architectures. |
| [requirements.txt](https://github.com/clementpoiret/hsf_train/blob/master/requirements.txt) | The `requirements.txt` file specifies dependencies essential for running the HSF Train repository, indicating the project's reliance on libraries such as PyTorch, Lightning, and ONNX for deep learning, model training, serialization, and tracking experiments.                                                                                                                                                                                                                                                                                                                                                                                                                   |

</details>

<details closed><summary>conf</summary>

| File                                                                                   | Summary                                                                                                                                                                                                                                                         |
| ---                                                                                    | ---                                                                                                                                                                                                                                                             |
| [config.yaml](https://github.com/clementpoiret/hsf_train/blob/master/conf/config.yaml) | This `config.yaml` integrates core components for the `hsf_train` repository, setting defaults for datasets, model architecture, training parameters, and logging, ensuring streamlined configuration and modular adaptation within the project's architecture. |

</details>

<details closed><summary>conf.datasets</summary>

| File                                                                                                            | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| ---                                                                                                             | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [all.yaml](https://github.com/clementpoiret/hsf_train/blob/master/conf/datasets/all.yaml)                       | The `conf/datasets/all.yaml` file serves as a comprehensive configuration for dataset management, specifically tailored for hippocampal MRI data across various domains such as `hiplay`, `memodev`, and more. It outlines data paths, MRI patterns, label mappings, and operation parameters like batch size and train/test split ratios, ensuring datasets are standardized and efficiently processed within the hsf_train repository's architecture for training and fine-tuning neural network models on hippocampal segmentation tasks. |
| [custom_dataset.yaml](https://github.com/clementpoiret/hsf_train/blob/master/conf/datasets/custom_dataset.yaml) | This configuration within `hsf_train` repository specifies loading and processing details for a custom dataset, including paths, batch size, worker count, memory pinning, partitioning ratios, and MRI image-label pairings for hippocampus study, streamlining dataset integration in the ML model training process.                                                                                                                                                                                                                       |

</details>

<details closed><summary>conf.finetuning</summary>

| File                                                                                                | Summary                                                                                                                                                                                                                                                                                                         |
| ---                                                                                                 | ---                                                                                                                                                                                                                                                                                                             |
| [default.yaml](https://github.com/clementpoiret/hsf_train/blob/master/conf/finetuning/default.yaml) | This `default.yaml` within `conf/finetuning/` defines parameters for fine-tuning processes in the `hsf_train` repository, focusing on decoder models. It specifies the model depth, unfreeze frequency for layers, and output channels, directly impacting model adaptation and performance optimization tasks. |

</details>

<details closed><summary>conf.lightning</summary>

| File                                                                                               | Summary                                                                                                                                                                                                                                          |
| ---                                                                                                | ---                                                                                                                                                                                                                                              |
| [default.yaml](https://github.com/clementpoiret/hsf_train/blob/master/conf/lightning/default.yaml) | The `conf/lightning/default.yaml` configures the training environment for the hsf_train repository, setting GPU acceleration, automatic strategy selection, mixed precision, and training parameters including epochs and gradient accumulation. |

</details>

<details closed><summary>conf.logger</summary>

| File                                                                                        | Summary                                                                                                                                                                                  |
| ---                                                                                         | ---                                                                                                                                                                                      |
| [wandb.yaml](https://github.com/clementpoiret/hsf_train/blob/master/conf/logger/wandb.yaml) | This YAML config file for Weights & Biases (Wandb) is integral to the repository's logging framework, setting up structured experiment tracking and visualization within an ML pipeline. |

</details>

<details closed><summary>conf.models</summary>

| File                                                                                            | Summary                                                                                                                                                                                                |
| ---                                                                                             | ---                                                                                                                                                                                                    |
| [default.yaml](https://github.com/clementpoiret/hsf_train/blob/master/conf/models/default.yaml) | This configuration file defines default hyperparameters for a 3D Residual U-Net model within the repository's deep learning training framework, including architecture specifics and training options. |

</details>

<details closed><summary>hsftrain</summary>

| File                                                                                         | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| ---                                                                                          | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| [callbacks.py](https://github.com/clementpoiret/hsf_train/blob/master/hsftrain/callbacks.py) | The `callbacks.py` in `hsf_train` enables SparseML integration for optimizing model training and exporting to ONNX. It facilitates training with a specified SparseML recipe, ensuring compatibility with single optimizer setups and supports model finalization and ONNX export with customizable settings for batch normalization fusing and QAT conversion.                                                                                                                                                                      |
| [exporter.py](https://github.com/clementpoiret/hsf_train/blob/master/hsftrain/exporter.py)   | The `exporter.py` module provides functionality to transform PyTorch models into ONNX format, emphasizing support for various ONNX versions and manipulation of models for optimal export handling, such as disabling batch norm fusing and adjusting quantization settings for ONNX compatibility. It integrates with the broader architecture by using configurations from `config.yaml` and potentially affecting model training and evaluation processes through improved model interchangeability and deployment-ready formats. |
| [utils.py](https://github.com/clementpoiret/hsf_train/blob/master/hsftrain/utils.py)         | The `utils.py` within the `hsf_train` repository is a utility module providing file integrity verification and dynamic model fetching capabilities. It ensures downloaded models match their expected hashes, offering an automated mechanism for maintaining model version control and integrity within this machine learning framework. This process is vital for the seamless operation and reliability of model training and inference processes in the repository's architecture.                                               |

</details>

<details closed><summary>hsftrain.models</summary>

| File                                                                                                | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| ---                                                                                                 | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| [blocks.py](https://github.com/clementpoiret/hsf_train/blob/master/hsftrain/models/blocks.py)       | The `blocks.py` module within the `hsf_train` repository defines custom neural network blocks using PyTorch, central to building model architectures. This component is pivotal for model customization and optimization in the project's deep learning framework.                                                                                                                                                                                                                                                                       |
| [helpers.py](https://github.com/clementpoiret/hsf_train/blob/master/hsftrain/models/helpers.py)     | This `helpers.py` module provides utility functions for models in the `hsf_train` repository. It calculates learnable model parameters and dynamically computes feature numbers per level, aiding in model adaptability and configuration optimization within the repository's machine learning architecture.                                                                                                                                                                                                                            |
| [layers.py](https://github.com/clementpoiret/hsf_train/blob/master/hsftrain/models/layers.py)       | The `SwitchNorm3d` class in `hsftrain/models/layers.py` introduces an adaptive normalization layer for 3D data, capable of dynamically selecting the best normalization method (Instance, Layer, or Batch Normalization) during training, optimizing for diverse architectures within the parent repository's machine learning framework focused on handling volumetric data.                                                                                                                                                            |
| [losses.py](https://github.com/clementpoiret/hsf_train/blob/master/hsftrain/models/losses.py)       | This code snippet, part of the `hsf_train` repository's neural network model architecture, introduces advanced loss functions tailored for segmentation tasks. It directly borrows from `nnUNet`, featuring methods like `TverskyLoss`, `FocalTversky_loss`, and utility functions to process tensor operations. These are pivotal for enhancing model training by focusing on minimizing segmentation errors, integrating novel mechanisms like forgiving loss to adjust model sensitivity towards specific types of prediction errors. |
| [models.py](https://github.com/clementpoiret/hsf_train/blob/master/hsftrain/models/models.py)       | The `models.py` file is central to defining the machine learning model architecture and evaluation metrics within the `hsf_train` repository. It leverages the Lightning framework for model training lifecycle management and utilizes the pymia library for defining evaluation criteria. This file directly supports model experimentation and optimization by providing a structured model definition and evaluation mechanism, crucial for the repository's goal of fine-tuning and evaluating models efficiently.                  |
| [optimizer.py](https://github.com/clementpoiret/hsf_train/blob/master/hsftrain/models/optimizer.py) | The `optimizer.py` within `hsf_train` repository introduces the AdamP optimization algorithm, enhancing training stability and performance for deep learning models by integrating innovations like gradient centralization, adaptive gradient clipping, and optional adanorm adaptation for more controlled weight updates, fitting harmoniously into the model training framework directed by configurations in its parent architecture.                                                                                               |
| [scheduler.py](https://github.com/clementpoiret/hsf_train/blob/master/hsftrain/models/scheduler.py) | This code, part of the hsf_train repository, implements a learning rate scheduler with options for linear warmup and cosine annealing or linear decay. It manages learning rate adjustments over epochs to optimize training, crucial for the training pipeline's efficiency and efficacy in model performance.                                                                                                                                                                                                                          |
| [types.py](https://github.com/clementpoiret/hsf_train/blob/master/hsftrain/models/types.py)         | This code snippet, part of the `hsf_train` repository's `models` module, defines crucial type aliases and structures for handling parameters, optimization, and loss computation in model training. It supports flexible, type-safe operations across the training pipeline by standardizing parameter, state, and loss representations.                                                                                                                                                                                                 |

</details>

<details closed><summary>scripts</summary>

| File                                                                                              | Summary                                                                                                                                                                                                                                                                                                                                                   |
| ---                                                                                               | ---                                                                                                                                                                                                                                                                                                                                                       |
| [ckpt_to_onnx.py](https://github.com/clementpoiret/hsf_train/blob/master/scripts/ckpt_to_onnx.py) | This script converts trained segmentation models from PyTorch checkpoints to ONNX format, enabling interoperability and optimization for deployment in diverse environments. It utilizes model parameters and custom loss functions defined within the repository's architecture, emphasizing the integration with the broader machine learning pipeline. |

</details>

<details closed><summary>sparseml</summary>

| File                                                                                               | Summary                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ---                                                                                                | ---                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| [finetuning.yaml](https://github.com/clementpoiret/hsf_train/blob/master/sparseml/finetuning.yaml) | The `sparseml/finetuning.yaml` serves as a configuration file within the hsf_train repository, outlining the finetuning phase's schedule. It specifies the number of training epochs, introduces pruning and quantization periods to optimize model size and inference speed, and defines the progression and intensity of model sparsity and quantization efforts. Its role is crucial for enhancing model performance and efficiency in the repository's overarching machine learning workflow. |
| [scratch.yaml](https://github.com/clementpoiret/hsf_train/blob/master/sparseml/scratch.yaml)       | The `sparseml/scratch.yaml` file defines a training schedule for neural network sparsity and quantization within the larger machine learning training framework, aiming for model efficiency. It schedules epoch-based pruning to reduce model size before applying quantization techniques for further compression, crucial for deployment efficiency. This configuration impacts overall training, inferencing speed, and model deployability, especially in resource-constrained environments. |

</details>

---

##  Getting Started

***Requirements***

Ensure you have the following dependencies installed on your system:

* **Python**: `version 3.8.0 or higher`

###  Installation

1. Clone the hsf_train repository:

```sh
git clone https://github.com/clementpoiret/hsf_train
```

2. Change to the project directory:

```sh
cd hsf_train
```

3. Install the dependencies:

```sh
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

4. Set up the environment variables for Weights & Biases:

```sh
touch .env
echo "WANDB_API_KEY=your_api_key" > .env
```

###  Running `hsf_train`

Use the following command to run train a new model:

```sh
python main.py
```

Example command to fine-tune a model:

```sh
python finetune.py datasets=custom_dataset finetuning.out_channels=8 models.lr=1e-3 models.use_forgiving_loss=False
```

---

##  Changelog

The implementation of the `hsf_train` repository is slightly different for the original paper as it proposes several improvements and optimizations.

Here are the main changes:

- All downsampling operations are switched from max-pooling to strided convolutions,
- ReLU activation functions are replaced by GELU,
- The model is trained with another optimizer, AdamP, which is a variant of Adam,
- Beyond AdamP, the learning rate is scheduled using a cosine annealing with warmup,
- AdamP is further improved with [Adaptive Gradient Clipping](https://arxiv.org/abs/2102.06171) and [Demon](https://arxiv.org/abs/1910.04952).

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **Submit Pull Requests**: Review open PRs, and submit your own PRs.
- **Report Issues**: Submit bugs found or log feature requests for the `hsf_train` project.

<details closed>
    <summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your github account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone https://github.com/clementpoiret/hsf_train
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to GitHub**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.

Once your PR is reviewed and approved, it will be merged into the main branch.

</details>

---

##  License

This project is protected under the MIT License.

---

## Citing HSF

If you use HSF in your research, please cite the following paper:

```
@ARTICLE{10.3389/fninf.2023.1130845,
    AUTHOR={Poiret, Clement and Bouyeure, Antoine and Patil, Sandesh and Grigis, Antoine and Duchesnay, Edouard and Faillot, Matthieu and Bottlaender, Michel and Lemaitre, Frederic and Noulhiane, Marion},
    TITLE={A fast and robust hippocampal subfields segmentation: HSF revealing lifespan volumetric dynamics},
	JOURNAL={Frontiers in Neuroinformatics},
	VOLUME={17},
	YEAR={2023},
	URL={https://www.frontiersin.org/articles/10.3389/fninf.2023.1130845},
	DOI={10.3389/fninf.2023.1130845},
	ISSN={1662-5196},
    ABSTRACT={The hippocampal subfields, pivotal to episodic memory, are distinct both in terms of cyto- and myeloarchitectony. Studying the structure of hippocampal subfields in vivo is crucial to understand volumetric trajectories across the lifespan, from the emergence of episodic memory during early childhood to memory impairments found in older adults. However, segmenting hippocampal subfields on conventional MRI sequences is challenging because of their small size. Furthermore, there is to date no unified segmentation protocol for the hippocampal subfields, which limits comparisons between studies. Therefore, we introduced a novel segmentation tool called HSF short for hippocampal segmentation factory, which leverages an end-to-end deep learning pipeline. First, we validated HSF against currently used tools (ASHS, HIPS, and HippUnfold). Then, we used HSF on 3,750 subjects from the HCP development, young adults, and aging datasets to study the effect of age and sex on hippocampal subfields volumes. Firstly, we showed HSF to be closer to manual segmentation than other currently used tools (p < 0.001), regarding the Dice Coefficient, Hausdorff Distance, and Volumetric Similarity. Then, we showed differential maturation and aging across subfields, with the dentate gyrus being the most affected by age. We also found faster growth and decay in men than in women for most hippocampal subfields. Thus, while we introduced a new, fast and robust end-to-end segmentation tool, our neuroanatomical results concerning the lifespan trajectories of the hippocampal subfields reconcile previous conflicting results.}
}
```
