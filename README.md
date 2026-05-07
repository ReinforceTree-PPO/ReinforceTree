# ReinforceTree: PPO-Guided Adaptive LoRA Routing for Continual Learning


This repository contains a PyTorch implementation of **ReinforceTree**, which introduces a **PPO-based dynamic routing policy** for continual learning. Method details, theoretical analysis, and experimental results are described in the accompanying paper.

## Code Structure
```
.
├── data/                         # Data directory for LLM-CL-Benchmark
├── model/                        # Model implementations
│   ├── Regular/                  # Regular model implementations
│   │   └── Tree_LoRA.py          # Main CL trainer (train_one_task loop)
│   ├── Dynamic_network/          # Dynamic network implementations
│   └── Replay/                   # Replay-based methods
├── training/                     # Training related code
│   ├── main.py                   # Entry point, arg parsing, DeepSpeed init
│   └── params.py                 # Method registry and dataset names
├── utils/                        # Utility functions
│   ├── data/                     # Data processing utilities
│   ├── flash_attention/          # Flash attention implementation
│   ├── my_peft/                  # Custom PEFT implementations
│   ├── kd_lora_tree.py           # KD-tree + PPO integration (core)
│   └── ppo_task_policy.py        # TaskSelectionPolicy, PPOBuffer, ppo_update
├── inference/                    # Inference related code
│   ├── infer_multi_command.py    # Multi-task inference script
│   └── collect_results.py        # Aggregates per-task evaluation
├── scripts/                      # Training and evaluation scripts
│   ├── lora_based_methods/
│   │   └── ReinforceTree.sh          # Main training script (supports --use_ppo)
│   └── run_all_exps.sh           # Run all experiments
├── evaluations/                  # Per-task evaluation scripts
└── requirements.txt              # Python dependencies
```

## Setup

The main dependencies are listed below. For a complete list, see `requirements.txt`:

```
torch==2.4.1
transformers==4.45.2
deepspeed==0.15.3
accelerate==1.0.1
peft==0.13.2
bitsandbytes==0.42.0
```

### Installation

```bash
python3.10 -m venv reinforcetree
source reinforcetree/bin/activate
pip install -r requirements.txt
```

### Data and Model Preparation

#### 1. LLM-CL-Benchmark

-   1. Extract the dataset in the `data/LLM-CL-Benchmark` directory. Our benchmark includes 24 different tasks, a mixing of [TRACE-LLM](https://github.com/BeyonderXX/TRACE) and the datasets used in [O-LoRA](https://github.com/cmnfriend/O-LoRA). Specifically, the tasks are:

        | C-STANCE        |  NumGLUE-cm   |      QQP      |
        | :-------------- | :-----------: | :-----------: |
        | **NumGLUE-ds**  |  **MultiRC**  |    **RTE**    |
        | **yelp**        | **ScienceQA** |  **amazon**   |
        | **MeetingBank** |   **FOMC**    |   **Lima**    |
        | **BoolQA**      |    **CB**     |   **Py150**   |
        | **dbpedia**     |    **WiC**    |   **yahoo**   |
        | **IMDB**        |   **MNLI**    | **20Minuten** |
        | **agnews**      |   **COPA**    |   **SST-2**   |

-   2. Download the pre-trained model from HuggingFace and place it in the `./PTM/` directory. e.g., for Mistral-7B-Instruct-v0.3:

        ```bash
        cd ./PTM
        git clone https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
        ```

Supported backbones in the paper:

| Model | Hugging Face Identifier |
| :--- | :--- |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` |
| LLaMA-2-7B-Chat | `meta-llama/Llama-2-7b-chat-hf` |
| Gemma-2B-it | `google/gemma-2b-it` |
| LLaMA-3.2-1B-Instruct | `meta-llama/Llama-3.2-1B-Instruct` |

Note: LLaMA models require accepting Meta's license on Hugging Face before downloading.

## Training and Evaluation

### Run ReinforceTree

```bash
export model_name="Mistral-7B-Instruct-v0.3"
bash scripts/lora_based_methods/Tree_LoRA.sh
```

The script loops over `lora_depth` values `[8, 16, 32, 64]` and runs train → inference → evaluation for each.

Key parameters in the training script:

-   `--use_ppo`: Enable PPO-based task selection (default: `True`)
-   `--ppo_lr`: Learning rate for PPO policy (default: `3e-4`)
-   `--ppo_clip`: PPO clipping epsilon ε (default: `0.2`)
-   `--reg`: Regularization weight λ (default: `0.5`)
-   `--lora_depth`: Number of LoRA layers in the tree (default: `8`)
-   `--data_path`: Path to the training dataset
-   `--dataset_name`: Names of the datasets to train on
-   `--num_train_epochs`: Number of training epochs per task

Or simply, run `./scripts/run_all_exps.sh` to run all the experiments.

### Inference & Evaluation

```bash
python inference/infer_multi_command.py \
    --model_name_or_path ./PTM/Mistral-7B-Instruct-v0.3 \
    --inference_model_path ./outputs_LLM-CL/<run_dir> \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --CL_method Tree_LoRA

python inference/collect_results.py \
    --inference_tasks C-STANCE,FOMC,MeetingBank,Py150,ScienceQA,NumGLUE-cm,NumGLUE-ds,20Minuten \
    --data_path ./outputs_LLM-CL/<run_dir>/predictions
```

Results are saved to `final_result.txt` in the predictions directory.

## Baseline Methods

We compare ReinforceTree against the following continual learning baselines used in the paper:

- **TreeLoRA** ([Qian et al., ICML 2025](https://github.com/ZinYY/TreeLoRA)) — Hierarchical gradient-similarity tree over LoRA adapters with LCB-based bandit selection. Our primary baseline.
- **HiDeLoRA** ([Wang et al., CVPR 2024](https://github.com/thu-ml/HiDe-Prompt)) — LoRA-based variant implemented within the HiDe-Prompt codebase.
- **O-LoRA** ([Wang et al., EMNLP 2023](https://github.com/cmnfriend/O-LoRA)) — Orthogonal Low-Rank Adaptation enforcing subspace orthogonality.
- **L2P** ([Wang et al., CVPR 2022](https://github.com/google-research/l2p)) — Learning to Prompt for continual learning.
- **DualPrompt** ([Wang et al., ECCV 2022](https://github.com/google-research/l2p)) — Complementary prompts for rehearsal-free continual learning.
- **EWC** ([Kirkpatrick et al., PNAS 2017](https://github.com/GMvandeVen/continual-learning)) — Elastic Weight Consolidation using Fisher information.
- **OGD** ([Farajtabar et al., AISTATS 2020](https://github.com/GMvandeVen/continual-learning)) — Orthogonal Gradient Descent for continual learning.
- **GEM** ([Lopez-Paz & Ranzato, NeurIPS 2017](https://github.com/facebookresearch/GradientEpisodicMemory)) — Gradient Episodic Memory.

## Citation

```bibtex
@misc{NeurIPS'26:ReinforceTree,
    author = {Anonymous Authors},
    title  = {ReinforceTree: PPO-Guided Adaptive LoRA Routing for Continual Learning},
    year   = {2026},
    note   = {Under review at NeurIPS 2026}
}
```
The citation entry will be updated to the official version upon publication.
