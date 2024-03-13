<h1 align="center">FastV</h1>

<p align="center">
<a href="https://arxiv.org/abs/2403.06764">
<img alt="Static Badge" src="https://img.shields.io/badge/FastV-ArXiv-red"></a>

<a href="https://arxiv.org/abs/2403.06764](https://www.fastv.work/">
<img alt="Static Badge" src="https://img.shields.io/badge/Demo-Gradio-yellow"></a>
</p>

*FastV is a plug-and-play inference acceleration method for large vision language models relying on visual tokens. It could reach 45\% theoretical FLOPs reduction without harming the performance through pruning redundant visual tokens in deep layers.*

<div align=center>
<img width="600" src="./figs/fastv_tradeoff.png"/>
</div>

---
*Scheduled Updates🔥*

1. - [x] Visualization [Online Demo](https://www.fastv.work/)
2. - [x] LVLM Inefficent Visual Attention Visualization Code
3. - [x] FastV Inference and Evaluation
4. - [ ] Integrate FastV to LLM inference framework

Stay tuned!

## 0. Setup
```bash
conda create -n fastv python=3.10
conda activate fastv
cd src
bash setup.sh
```


## 1. Online Demo

We provide an [online demo](https://www.fastv.work/) for the FastV model. You can upload an image, enter a prompt, and select the number of layers to get the generated response and visualize the attention maps.

If you want to start your own demo, run the following script:
```bash
python demo.py --model-path ./llava-v1.5-7b
```

## 2. Visualization: Inefficient Attention over Visual Tokens 

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

## 3. FastV Inference and Evaluation

We provide code to reproduce the ablation study on K and R values, as shown in figure-7 in the paper. This implementation masks out the discarded tokens in deep layers for convenience. Inplace token dropping feature would be added in LLM inference framework section.

*ocrvqa*
```bash
bash ./src/FastV/inference/eval/eval_ocrvqa.sh
```

## 4. Integrate FastV to LLM inference framework

*Stay Tuned! Welcome discussion and contribution!*


<div align=center>
<img width="300" src="./figs/ablation_ocrvqa.png"/><br>
Results 
</div>


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
