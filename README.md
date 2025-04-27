# Expanding Scene Graph Boundaries: Fully Open-vocabulary Scene Graph Generation via Visual-Concept Alignment and Retention

[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2311.10988-b31b1b.svg)](https://arxiv.org/pdf/2311.10988)

**Official Implementation of**  
**"Expanding Scene Graph Boundaries: Fully Open-vocabulary Scene Graph Generation via Visual-Concept Alignment and Retention"**  
🏆 *Recognized as "Best Paper Candidate" at ECCV 2024 (Milan, Italy)*

---

![OvSGG](figures/OvSGG.png)
![OvSGTR](figures/OvSGTR.png)

---

## 📰 News
- [x] 2025.02: Add checkpoints for the TPAMI version
- [x] 2024.10: Our paper has been recognized as **"Best Paper Candidate"** (Milan, Italy, ECCV 2024)

---

## 🛠️ Setup

For simplicity, you can directly run:

```bash
bash install.sh
```

which includes the following steps:

0. Install PyTorch 1.9.1 and other dependencies:

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```
*(Adjust CUDA version if necessary.)*

1. Install GroundingDINO and download pretrained weights:

```bash
cd GroundingDINO && python3 setup.py install
mkdir $PWD/GroundingDINO/weights/
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -O $PWD/GroundingDINO/weights/groundingdino_swint_ogc.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth -O $PWD/GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
```

---

## 📚 Dataset

Supported datasets:
- VG150
- COCO

Prepare the dataset under `data/` folder following the [instruction](datasets/data.md).

---

## 📈 Closed-set SGG

### Training

```bash
bash scripts/DINO_train_dist.sh vg ./config/GroundingDINO_SwinT_OGC_full.py ./data ./logs/ovsgtr_vg_swint_full ./GroundingDINO/weights/groundingdino_swint_ogc.pth
```

or using Swin-B:

```bash
bash scripts/DINO_train_dist.sh vg ./config/GroundingDINO_SwinB_full.py ./data ./logs/ovsgtr_vg_swinb_full ./GroundingDINO/weights/groundingdino_swinb_cogcoor.pth
```

> Adjust `CUDA_VISIBLE_DEVICES` if needed. Effective batch size = batch size × number of GPUs.

### Inference

```bash
bash scripts/DINO_eval.sh vg [config file] [data path] [output path] [checkpoint]
```
or
```bash
bash scripts/DINO_eval_dist.sh vg [config file] [data path] [output path] [checkpoint]
```

---

![Benchmark on Closed-set SGG](figures/closed-sgg.png)

---

## 📥 Checkpoints (Closed-set SGG)

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@20/50/100</th>
      <th>Checkpoint</th>
      <th>Config</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Swin-T</td>
      <td>26.97 / 35.82 / 41.38</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-swint-full.pth">link</a></td>
      <td>config/GroundingDINO_SwinT_OGC_full.py</td>
    </tr>
    <tr>
      <td>Swin-T (pretrained on MegaSG)</td>
      <td>27.34 / 36.27 / 41.95</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-full-swint-mega-best.pth">link</a></td>
      <td>config/GroundingDINO_SwinT_OGC_full.py</td>
    </tr>
    <tr>
      <td>Swin-B</td>
      <td>27.75 / 36.44 / 42.35</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-swinb-full.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_full.py</td>
    </tr>
    <tr>
      <td>Swin-B (w/o freq bias & focal loss)</td>
      <td>27.53 / 36.18 / 41.79</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-swinb-full-open.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_full_open.py</td>
    </tr>
    <tr>
      <td>Swin-B (pretrained on MegaSG)</td>
      <td>28.61 / 37.58 / 43.41</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-full-swinb-mega-best.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_full_open.py</td>
    </tr>
  </tbody>
</table>

---

## 🚀 OvD-SGG (Open-vocabulary Detection SGG)

Set:

```python
sg_ovd_mode = True
```

---

### 📥 Checkpoints (OvD-SGG)

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@20/50/100 (Base+Novel)</th>
      <th>R@20/50/100 (Novel)</th>
      <th>Checkpoint</th>
      <th>Config</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Swin-T</td>
      <td>12.34 / 18.14 / 23.20</td>
      <td>6.90 / 12.06 / 16.49</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovd-swint.pth">link</a></td>
      <td>config/GroundingDINO_SwinT_OGC_ovd.py</td>
    </tr>
    <tr>
      <td>Swin-B</td>
      <td>15.43 / 21.35 / 26.22</td>
      <td>10.21 / 15.58 / 19.96</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovd-swinb.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_ovd.py</td>
    </tr>
    <tr>
      <td>Swin-T (pretrained on MegaSG)</td>
      <td>14.33 / 20.91 / 25.98</td>
      <td>10.52 / 17.30 / 22.90</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovd-swint-mega-best.pth">link</a></td>
      <td>config/GroundingDINO_SwinT_OGC_ovd.py</td>
    </tr>
    <tr>
      <td>Swin-B (pretrained on MegaSG)</td>
      <td>15.21 / 21.21 / 26.12</td>
      <td>10.31 / 15.78 / 20.47</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovd-swinb-mega-best.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_ovd.py</td>
    </tr>
  </tbody>
</table>

---
---

## 🔥 OvR-SGG (Open-vocabulary Relation SGG)

Set:

```python
sg_ovr_mode = True
```

---

### 📥 Checkpoints (OvR-SGG)

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@20/50/100 (Base+Novel)</th>
      <th>R@20/50/100 (Novel)</th>
      <th>Checkpoint</th>
      <th>Config</th>
      <th>Pre-trained Checkpoint</th>
      <th>Pre-trained Config</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Swin-T</td>
      <td>15.85 / 20.50 / 23.90</td>
      <td>10.17 / 13.47 / 16.20</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovr-swint.pth">link</a></td>
      <td>config/GroundingDINO_SwinT_OGC_ovr.py</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-pretrain-coco-swint.pth"><s>link</s></a></td>
      <td>config/GroundingDINO_SwinT_OGC_pretrain.py</td>
    </tr>
    <tr>
      <td>Swin-B</td>
      <td>17.63 / 22.90 / 26.68</td>
      <td>12.09 / 16.37 / 19.73</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovr-swinb.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_ovr.py</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-pretrain-coco-swinb.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_pretrain.py</td>
    </tr>
    <tr>
      <td>Swin-T (pretrained on MegaSG)</td>
      <td>19.38 / 25.40 / 29.71</td>
      <td>12.23 / 17.02 / 21.15</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovr-swint-mega-best.pth">link</a></td>
      <td>config/GroundingDINO_SwinT_OGC_ovr.py</td>
      <td><s>link</s></td>
      <td>config/GroundingDINO_SwinT_OGC_pretrain.py</td>
    </tr>
    <tr>
      <td>Swin-B (pretrained on MegaSG)</td>
      <td>21.09 / 27.92 / 32.74</td>
      <td>16.59 / 22.86 / 27.73</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovr-swinb-mega-best.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_ovr.py</td>
      <td><s>link</s></td>
      <td>config/GroundingDINO_SwinB_pretrain.py</td>
    </tr>
  </tbody>
</table>

---

## 🌟 OvD+R-SGG (Joint Open-vocabulary SGG)

Set:

```python
sg_ovd_mode = True
sg_ovr_mode = True
```

---

### 📥 Checkpoints (OvD+R-SGG)

<table>
  <thead>
    <tr>
      <th>Backbone</th>
      <th>R@20/50/100 (Joint)</th>
      <th>R@20/50/100 (Novel Object)</th>
      <th>R@20/50/100 (Novel Relation)</th>
      <th>Checkpoint</th>
      <th>Config</th>
      <th>Pre-trained Checkpoint</th>
      <th>Pre-trained Config</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Swin-T</td>
      <td>10.02 / 13.50 / 16.37</td>
      <td>10.56 / 14.32 / 17.48</td>
      <td>7.09 / 9.19 / 11.18</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovdr-swint.pth">link</a></td>
      <td>config/GroundingDINO_SwinT_OGC_ovdr.py</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-pretrain-coco-swint.pth"><s>link</s></a></td>
      <td>config/GroundingDINO_SwinT_OGC_pretrain.py</td>
    </tr>
    <tr>
      <td>Swin-B</td>
      <td>12.37 / 17.14 / 21.03</td>
      <td>12.63 / 17.58 / 21.70</td>
      <td>10.56 / 14.62 / 18.22</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovdr-swinb.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_ovdr.py</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-pretrain-coco-swinb.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_pretrain.py</td>
    </tr>
    <tr>
      <td>Swin-T (pretrained on MegaSG)</td>
      <td>10.67 / 15.15 / 18.82</td>
      <td>8.22 / 12.49 / 16.29</td>
      <td>9.62 / 13.68 / 17.19</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovdr-swint-mega-best.pth">link</a></td>
      <td>config/GroundingDINO_SwinT_OGC_ovdr.py</td>
      <td><s>link</s></td>
      <td>config/GroundingDINO_SwinT_OGC_pretrain.py</td>
    </tr>
    <tr>
      <td>Swin-B (pretrained on MegaSG)</td>
      <td>12.54 / 17.84 / 21.95</td>
      <td>10.29 / 15.66 / 19.84</td>
      <td>12.21 / 17.15 / 21.05</td>
      <td><a href="https://huggingface.co/JosephZ/OvSGTR/blob/main/vg-ovdr-swinb-mega-best.pth">link</a></td>
      <td>config/GroundingDINO_SwinB_ovdr.py</td>
      <td><s>link</s></td>
      <td>config/GroundingDINO_SwinB_pretrain.py</td>
    </tr>
  </tbody>
</table>

---

## 🤝 Acknowledgement

We thank:

- [Scene-Graph-Benchmark.pytorch](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO)

for their awesome open-source codes and models.

---

## 📖 Citation

If you find our work helpful, please cite:

```bibtex
@inproceedings{chen2024expanding,
  title={Expanding Scene Graph Boundaries: Fully Open-vocabulary Scene Graph Generation via Visual-Concept Alignment and Retention},
  author={Chen, Zuyao and Wu, Jinlin and Lei, Zhen and Zhang, Zhaoxiang and Chen, Changwen},
  booktitle={European Conference on Computer Vision (ECCV)},
  pages={108--124},
  year={2024}
}
```

---

# ✨ Enjoy Exploring Open-Vocabulary Scene Graph Generation!
