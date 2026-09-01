# NFC_VIT

Implementation of the paper:

> *Vision Transformer assisting rheumatologists in screening for capillaroscopy changes in systemic sclerosis: an artificial intelligence model*

Repository: [uzh-dqbm-cmi/NFC_VIT](https://github.com/uzh-dqbm-cmi/NFC_VIT)

## Model

Multi-task Vision Transformer (`ViT-Large`, patch size 32, input **384×384**) predicting severity (`0` / `+` / `++` / `+++`) for:

| Task key | Finding |
| --- | --- |
| `finger_dilatierte` | enlarged capillaries |
| `finger_riesen` | giant capillaries |
| `finger_rare` | capillary loss |
| `finger_mikro` | microhaemorrhages |

### Checkpoints (cluster)

```
/cluster/dataset/medinfmk/capillaroscopy/Nail-Imaging/multi-task/multiTask/{0-4}/pytorch_model.bin
```

Related data on the cluster:

```
/cluster/dataset/medinfmk/capillaroscopy/Nail-Imaging/multi-task/   # predictions + MultiTask_results.ipynb
/cluster/dataset/medinfmk/capillaroscopy/content/images/            # images
```

## Setup

Python **3.9** recommended.

```bash
git clone https://github.com/uzh-dqbm-cmi/NFC_VIT.git
cd NFC_VIT
git checkout add-infer-script   # until merged into main

conda create -n nail-imaging python=3.9 -y
conda activate nail-imaging
pip install -r requirements.txt
```

GPU (CUDA 11.3) — install torch first, then requirements:

```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 \
  -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

## Inference (single image)

```bash
python infer.py \
  --image /cluster/dataset/medinfmk/capillaroscopy/content/images/<one_image>.jpg \
  --checkpoint /cluster/dataset/medinfmk/capillaroscopy/Nail-Imaging/multi-task/multiTask/0/pytorch_model.bin
```

Optional: `--device cuda`

`infer.py` instantiates the model with `pretrained=False` and loads the fine-tuned `pytorch_model.bin` (no need for the ImageNet init weights).

## Training / evaluation

See `train.py`, `job.sh`, and `job-eval.sh`. Example multi-task training flags:

```bash
python train.py \
  --data_dir /cluster/dataset/medinfmk/capillaroscopy/ \
  --output_dir <output_dir> \
  --do_train --do_eval \
  --multiTask \
  --task_name NailImages \
  ...
```

## Repository layout

| Path | Description |
| --- | --- |
| `infer.py` | Single-image inference |
| `requirements.txt` | Pinned inference dependencies |
| `Model/` | Model, datasets, processors |
| `train.py` | Training / CV evaluation |
| `job.sh` / `job-eval.sh` | Cluster job examples |

## Example notebook (one image per class)

Run or view the executed demo:

- [`infer_examples.ipynb`](infer_examples.ipynb) — picks one fold-0 test image per severity class for each task and compares GT vs prediction
- Outputs: [`infer_examples_outputs/`](infer_examples_outputs/) (`summary.csv` + per-task PNGs)
