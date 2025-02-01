<div align="center">
  <h2><b>Poison as Cure: Visual Noise for Mitigating Object Hallucinations in LVMs</b></h2>
  <h4>What doesn't kill me makes me stronger</h4>
</div>

<div align="center">
  <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/KejiaZhang-Robust/VAP">
  <img alt="GitHub forks" src="https://img.shields.io/github/forks/KejiaZhang-Robust/VAP">
  <a href="https://arxiv.org/abs/2412.00143">
    <img src="https://img.shields.io/badge/arXiv-2412.00143-b31b1b" alt="arXiv" />
  </a>
  <img alt="GitHub License" src="https://img.shields.io/github/license/KejiaZhang-Robust/VAP">
</div>

<div align="center">
  <img src="image/westlake_signatures.png" height="100" alt="Westlake University Logo" style="margin-right: 20px;">
  <img src="image/DAMO.avif" height="100" alt="DAMO Logo" style="margin-right: 20px;">
</div>

## Dataset

[BEAF](https://drive.google.com/file/d/1Xx7j8Hz8QX3Fl_hpSBet6r15njhwCgeR/view)

[Pope](https://github.com/RUCAIBox/POPE?tab=readme-ov-file)

## Running Code

1. LVMs VQA Inference

```
sh script/VQA.sh
```

2. LVMs VQA Inference under VAP vision input.

```
sh script/VAP.sh
```

3. Evaluate Hallucination

```
sh script/evaluate.sh
```

## Setup Environment

### LLAVA-v1.5

- liuhaotian/llava-v1.5-7b [Huggingface Page](https://huggingface.co/liuhaotian/llava-v1.5-7b)

```
cd env_setting/LLaVA
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip
pip install -e .
pip install ftfy regex tqdm
pip install protobuf
pip install transformers_stream_generator
pip install matplotlib
```

## Instruct-BLIP-7B

- Salesforce/instructblip-vicuna-7b [Huggingface Page](https://huggingface.co/Salesforce/instructblip-vicuna-7b)

## Qwen2-vl

- Qwen/Qwen2-VL-7B-Instruct: [Huggingface Page](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)

```
conda create -n qwen python==3.11 -y
conda activate qwen
cd env_setting/transformers
pip install .
pip install qwen-vl-utils
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install accelerate==0.26.1
pip install ftfy regex tqdm
pip install matplotlib
```

## InternVL Series

- OpenGVLab/InternVL2-8B [Huggingface Page](https://huggingface.co/OpenGVLab/InternVL2-8B)
- OpenGVLab/InternVL2-8B-MPO [Huggingface Page](https://huggingface.co/OpenGVLab/InternVL2-8B-MPO)

```
conda create -n internvl python=3.9 -y
conda activate internvl
pip install lmdeploy==0.5.3
pip install timm
pip install ftfy regex tqdm
pip install matplotlib
pip install flash-attn==2.3.6 --no-build-isolation
```

## LLaVA-OneVision-7B

- lmms-lab/llava-onevision-qwen2-7b-ov [Huggingface Page](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov)

```
cd env_setting/LLaVA-NeXT
conda create -n llava_onevision python=3.10 -y
conda activate llava_onevision
pip install --upgrade pip
pip install -e ".[train]"
pip install ftfy regex tqdm
cd ../transformers
pip install .
pip install transformers==4.47.0
pip install flash_attn-2.5.2+cu122torch2.1cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install matplotlib
```

## deepseek-vl2

- deepseek-ai/deepseek-vl2 [Huggingface Page](https://huggingface.co/deepseek-ai/deepseek-vl2)

```
conda create -n deepseek python==3.10 -y
conda activate deepseek
cd env_setting/DeepSeek-VL2/
pip install -e .
pip install ftfy regex tqdm
pip install matplotlib
pip install flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install --force-reinstall --no-deps --pre xformers
pip install transformers==4.38.2
```
