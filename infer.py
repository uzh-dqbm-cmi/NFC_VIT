"""
Single-image inference for the multi-task ViT capillaroscopy model.

Example:
  python infer.py \
    --image /cluster/dataset/medinfmk/capillaroscopy/content/images/<one_image>.jpg \
    --checkpoint /cluster/dataset/medinfmk/capillaroscopy/Nail-Imaging/multi-task/multiTask/0/pytorch_model.bin
"""

import argparse
import os
import sys

import torch
from PIL import Image
from torchvision import transforms

# Allow running from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import Model.viTransformer as vt
from Model.classifier import MultiTaskClassificationModel

# Do not download / load ImageNet init weights; we load the fine-tuned checkpoint.
_orig_vit = vt.vit_large_patch32_384


def _vit_no_pretrained(*args, **kwargs):
    kwargs["pretrained"] = False
    return _orig_vit(*args, **kwargs)


vt.vit_large_patch32_384 = _vit_no_pretrained

TASKS = [
    "finger_dilatierte",  # enlarged capillaries
    "finger_riesen",      # giant capillaries
    "finger_rare",        # capillary loss
    "finger_mikro",       # microhaemorrhages
]
TASK_DISPLAY = {
    "finger_dilatierte": "enlarged capillaries",
    "finger_riesen": "giant capillaries",
    "finger_rare": "capillary loss",
    "finger_mikro": "microhaemorrhages",
}
SEVERITY = {0: "0", 1: "+", 2: "++", 3: "+++"}
NUM_LABELS_PER_TASK = {t: 4 for t in TASKS}

IMAGENET_RGB_MEAN = [0.485, 0.456, 0.406]
IMAGENET_RGB_SD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((384, 384)),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_RGB_MEAN, IMAGENET_RGB_SD),
    ]
)


def load_model(checkpoint_path: str, device: torch.device) -> MultiTaskClassificationModel:
    model = MultiTaskClassificationModel(num_labels_per_task=NUM_LABELS_PER_TASK)
    state = torch.load(checkpoint_path, map_location=device)
    # Support both bare state_dict and {"state_dict": ...}
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model_dict = model.state_dict()
    matched = {k: v for k, v in state.items() if k in model_dict and model_dict[k].shape == v.shape}
    missing = [k for k in model_dict if k not in matched]
    if missing:
        raise RuntimeError(f"Checkpoint missing {len(missing)} keys, e.g. {missing[:5]}")
    model_dict.update(matched)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    return model


def predict_one(model: MultiTaskClassificationModel, image_path: str, device: torch.device):
    image = Image.open(image_path).convert("RGB")
    x = TRANSFORM(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model.visual_features(x)
    results = {}
    for task, logit in logits.items():
        probs = torch.softmax(logit, dim=-1)[0]
        pred_id = int(torch.argmax(probs).item())
        results[task] = {
            "label": SEVERITY[pred_id],
            "class_id": pred_id,
            "probs": {SEVERITY[i]: float(probs[i]) for i in range(4)},
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Run multi-task ViT inference on one image")
    parser.add_argument("--image", required=True, help="Path to a single capillaroscopy image")
    parser.add_argument(
        "--checkpoint",
        default="/cluster/dataset/medinfmk/capillaroscopy/Nail-Imaging/multi-task/multiTask/0/pytorch_model.bin",
        help="Path to pytorch_model.bin (default: fold 0 on cluster)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image: {args.image}")

    model = load_model(args.checkpoint, device)
    results = predict_one(model, args.image, device)

    print("\nPredictions:")
    for task in TASKS:
        r = results[task]
        print(f"  {TASK_DISPLAY[task]:24s} ({task}): {r['label']}")
        probs_str = ", ".join(f"{k}={v:.3f}" for k, v in r["probs"].items())
        print(f"    probs: {probs_str}")


if __name__ == "__main__":
    main()
