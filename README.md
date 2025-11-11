# PAMR: Persistent Autoregressive Mapping with Traffic Rules

<h3 align="center"><strong>🎉🎉This paper has been accepted by AAAI 2026🎉🎉</strong></h3>
<a href="https://arxiv.org/abs/2509.22756"><img src='https://img.shields.io/badge/arXiv-Paper-red?logo=arxiv&logoColor=white' alt='arXiv'></a>
<a href='https://miv-xjtu.github.io/PAMR.github.io'><img src='https://img.shields.io/badge/Project_Page-Website-green?logo=googlechrome&logoColor=white' alt='Project Page'></a>


Official implementation of the paper: **"Persistent Autoregressive Mapping with Traffic Rules for Autonomous Driving"**. This repository contains the source code for the PAMR framework, which performs autoregressive co-construction of lane vectors and traffic rules from visual observations.

<br>

> **Abstract:** Safe autonomous driving requires both accurate HD map construction and persistent awareness of traffic rules, even when their associated signs are no longer visible. However, existing methods either focus solely on geometric elements or treat rules as temporary classifications, failing to capture their persistent effectiveness across extended driving sequences. In this paper, we present PAMR (Persistent Autoregressive Mapping with Traffic Rules), a novel framework that performs autoregressive co-construction of lane vectors and traffic rules from visual observations. Our approach introduces two key mechanisms: Map-Rule Co-Construction for processing driving scenes in temporal segments, and Map-Rule Cache for maintaining rule consistency across these segments. To properly evaluate continuous and consistent map generation, we develop MapDRv2, featuring improved lane geometry annotations. Extensive experiments demonstrate that PAMR achieves superior performance in joint vector-rule mapping tasks, while maintaining persistent rule effectiveness throughout extended driving sequences.

<br>

## 🌟 Key Features

- **Persistent Autoregressive Mapping:** A novel framework that co-constructs lane vectors and their associated traffic rules in an autoregressive manner, mimicking how a human driver "narrates" a road scene.
- **Map-Rule Cache & Co-Construction:** A mechanism to process driving scenes in temporal segments and propagate semantic and geometric information, ensuring rule persistence even when signs are no longer visible.
- **Introduces MapDRv2:** A re-annotated version of the MapDR dataset with smooth, continuous lane geometries, providing a robust benchmark for evaluating continuous map generation.
- **Interactive Prompting:** Allows for targeted map construction by prompting the model with specific traffic sign locations via coordinates, bounding boxes, or visual ROIs.
- **End-to-End Framework:** Integrates geometric lane construction and semantic rule association into a single, unified MLLM-based pipeline.

## 🖼️ Framework Overview

The PAMR framework processes sequential visual and trajectory data to generate a coherent HD map. It operates in segments, using a **Map-Rule Cache** to ensure seamless propagation of information, enabling persistent rule awareness and continuous vector generation.

![Framework Overview](./assets/framework.png)
> **Left:** The sequential processing of map-rules. **Right:** A bird's-eye view visualization of the map-rule construction process, showing how the cache enables consistent map generation across segments.

## ⚙️ Installation

Create the required environment through the following steps:

```bash
git clone https://github.com/MIV-XJTU/PAMR.git && cd PAMR

conda create -n PAMR python=3.10 -y && conda activate PAMR

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
cd transformers && pip install -e .

cd ../
pip install ./qwen_vl_utils-0.0.8-py3-none-any.whl
pip install ./vllm-0.6.1-cp38-abi3-linux_x86_64.whl
pip install -r requirements.txt
```

## 📊 Dataset (MapDRv2)

This project is evaluated on the **MapDRv2** dataset, which features enhanced, continuous lane geometry annotations suitable for evaluating persistent mapping.

1.  **Download the Dataset:** Please follow the instructions from the official MapDR project to obtain access to the dataset.
2.  **Data Structure:** Organize the dataset as follows. The `--data_path` argument should point to the root directory (`/path/to/mapdr_v2`).

    ```
    /path/to/mapdr_v2/
    ├── <uid_1>/
    │   ├── data_v3.json
    │   ├── label_v3.json
    │   └── img/
    │       ├── <timestamp_1>.jpg
    │       ├── <timestamp_2>.jpg
    │       └── ...
    ├── <uid_2>/
    │   ├── ...
    └── ...
    ```


## 🧠 Model Weights

Before running the inference, you need to download the model weights **[Download from Hugging Face Hub](xxx)**.
<!-- 
    ```
    pamr/
    ├── network/
    │   └── qwen2-vl-2b-model/ 
    │       ├── config.json
    │       ├── modeling_qwen.py
    │       ├── pytorch_model.bin.index.json
    │       └── ...
    ├── README.md
    ├── src/
    └── ...
    ``` -->


## 🚀 Inference

The main script `src/infer_online_vvlm_mapdr_patch.py` handles both single-GPU debugging and multi-GPU parallel inference.

### Running Inference

Use the following command to run inference on the dataset:

```bash
# Set the GPUs you want to use
export CUDA_VISIBLE_DEVICES=0

python main.py \
    --model_path /path/to/model_weight \
    --output_path /path/to/save/outputs \
    --uid_list_file ./data_split/mapdr_v2_test.txt \
    --data_path /path/to/mapdr_v2 \
    --visualize \ # Optional: to save visualization images
    --debug \ # Optional: debug mode
```

## 📜 Citing

If you find PAMR is useful in your research or applications, please consider giving us a star 🌟 and citing it by the following BibTeX entry:

```
@article{liang2025persistent,
  title={Persistent Autoregressive Mapping with Traffic Rules for Autonomous Driving},
  author={Liang, Shiyi and Chang, Xinyuan and Wu, Changjie and Yan, Huiyuan and Bai, Yifan and Liu, Xinran and Zhang, Hang and Yuan, Yujian and Zeng, Shuang and Xu, Mu and others},
  journal={arXiv preprint arXiv:2509.22756},
  year={2025}
}
```

## 🙏 Acknowledgement
Our work is primarily based on the following codebases:[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [MAPDR](https://github.com/MIV-XJTU/MapDR). We are sincerely grateful for their work.

