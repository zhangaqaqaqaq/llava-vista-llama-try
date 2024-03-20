<h1 align="center">FastV</h1>

<p align="center">
<a href="https://arxiv.org/abs/2403.06764">
<img alt="Static Badge" src="https://img.shields.io/badge/FastV-ArXiv-red"></a>

<a href="https://www.fastv.work/">
<img alt="Static Badge" src="https://img.shields.io/badge/Demo-Gradio-yellow"></a>
</p>

*FastV is a plug-and-play inference acceleration method for large vision language models relying on visual tokens. It could reach 45\% theoretical FLOPs reduction without harming the performance through pruning redundant visual tokens in deep layers.*

<div align=center>
<img width="600" src="./figs/fastv_tradeoff.png"/>
</div>

---
*Scheduled Updates🔥*

0. - [x] Setup
1. - [x] Visualization [Online Demo](https://www.fastv.work/)
2. - [x] LVLM Inefficent Visual Attention Visualization Code
3. - [x] FastV Inference and Evaluation
4. - [x] Latency Test Reproduction
5. - [ ] Support KV Cache

Stay tuned!

## Setup
```bash
conda create -n fastv python=3.10
conda activate fastv
cd src
bash setup.sh
```


## Online Demo

We provide an [online demo](https://www.fastv.work/) for the FastV model. You can upload an image, enter a prompt, and select the number of layers to get the generated response and visualize the attention maps.

If you want to start your own demo, run the following script:
```bash
python demo.py --model-path ./llava-v1.5-7b
```

## Visualization: Inefficient Attention over Visual Tokens 

we provide a script (./src/FastV/inference/visualization.sh) to reproduce the visualization result of each LLaVA model layer for a given image and prompt.

```bash
bash ./src/FastV/inference/visualization.sh
```
or
```bash
python ./src/FastV/inference/plot_inefficient_attention.py \
    --model-path "PATH-to-HF-LLaVA1.5-Checkpoints" \
    --image-path "./src/LLaVA/images/llava_logo.png" \
    --prompt "Describe the image in details."\
    --output-path "./output_example"\
```

Model output and attention maps for different layers would be stored at "./output_example"

<div align=center>
<img width="600" src="./figs/attn_map.png"/>
</div>

## FastV Inference and Evaluation

We provide code to reproduce the ablation study on K and R values, as shown in figure-7 in the paper. This implementation masks out the discarded tokens (no speed up) in deep layers for convenience and fair performance comparison.

*ocrvqa*
```bash
bash ./src/FastV/inference/eval/eval_ocrvqa_fastv_token_mask
```

<div align=center>
<img width="300" src="./figs/ablation_ocrvqa.png"/><br>
Results 
</div>


## Integrate FastV to LLM inference framework


### Latency Experiment Reproduction
You could use following code to reproduce FastV's latency experiment on aokvqa. We conduct the following experiments on one A100 GPU (80G) 

```bash
bash ./src/FastV/inference/eval/eval_aokvqa_latency_fastv_inplace.sh
```

*aokvqa results*
| Model                         | Score | latency / first output token (A100 80G) | GPU Memory |
| ----------------------------- | ----- | --------------------------------------- | ---------- |
| 7B Vanilla Decoding           | 76.8  | 0.138s                                  | 18G        |
| 13B Vanilla Decoding          | 81.9  | 0.203s                                  | 33G        |
| \- 13B FastV (K=2 R=25%)      | 81.8  | 0.181s                                  | 29G        |
| \- 13B FastV (K=2 R=50%)      | 81.3  | 0.155s                                  | 28G        |
| \- 13B FastV (K=2 R=75%)      | 80.9  | **0.124s**                                  | 27G        |
| 13B Vanilla Decoding 4Bit     | 81.5  | 0.308s                                  | 12G        |
| \- 13B FastV 4Bit (K=2 R=25%) | 81.7  | 0.277s                                  | 11G        |
| \- 13B FastV 4Bit (K=2 R=50%) | 81.1  | 0.275s                                  | 10G        |
| \- 13B FastV 4Bit (K=2 R=75%) | 80.3  | 0.245s                                  | **9G**         |

This code implements the latency test of FastV using inplace token dropping instead of token masking (support K>0). It is not compatible with kv-cache yet, must be used with "use_cache=False" in the generate function. FastV is also compatible with model quantization, just set the 4bit/8bit flag to be true from [inference_aokvqa.py](https://github.com/pkunlp-icler/FastV/blob/main/src/FastV/inference/eval/inference_aokvqa.py#L192) to see the performance.

The main implementation of FastV is in the forward function of LlamaModel from [modeling_llama.py](https://github.com/pkunlp-icler/FastV/blob/main/src/transformers/src/transformers/models/llama/modeling_llama.py#L730) of transformers repo.

### Support KV Cache

Stay tuned!


## Citation
```bib
@misc{chen2024image,
      title={An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play Inference Acceleration for Large Vision-Language Models}, 
      author={Liang Chen and Haozhe Zhao and Tianyu Liu and Shuai Bai and Junyang Lin and Chang Zhou and Baobao Chang},
      year={2024},
      eprint={2403.06764},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
