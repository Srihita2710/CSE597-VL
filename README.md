# Surgical-VQLA++: Adversarial Contrastive Learning for Robust VQLA in Robotic Surgery

---

## Environment

This project uses the following libraries:
PyTorch, numpy, pandas, scipy, scikit-learn, timm, transformers, h5py

---

## Directory Structure

- `checkpoints/`: Trained model weights. 
- `dataset/`:
  - `bertvocab/v2`: BERT tokenizer.
  - `EndoVis-18-VQLA/`: Dataset for training and evaluation.
    - `seq_1` to `seq_16`:
      - `left_frames/`: Image frames (download from EndoVIS18 challenge).
      - `vqla/`:
        - `label/`: Q&A pairs with bounding box labels.
        - `img_features/`: Extracted image features:
          - `5x5/`: Features using ResNet18 with a patch size of 5x5.
          - `frcnn/`: Features using Fast-RCNN with ResNet101.
  - `EndoVis-17-VQLA/`: 97 validation frames from EndoVIS17 challenge with a similar structure as above.
- `models/`: Model implementations:
  - `CATViLEmbedding.py`: Proposed model for VQLA.
  - `DeiTPrediction.py`: DeiT encoder-based model.
  - `VisualBertResMLP.py`: VisualBERT ResMLP encoder from Surgical-VQA.
  - Additional prediction models.
- Key scripts:
  - `dataloader.py`: Data loading utilities.
  - `train.py`: Training and evaluation script.
  - `utils.py`: Helper functions.

---

## Dataset

[EndoVis17/18-VQLA-Extended](https://drive.google.com/file/d/1-FXOdhD3uw55ATDgI1wPEe-txyuCiP2E/view?usp=drive_link) put in dataset folder.

---

## Training

To train on the EndoVis-18-VQLA-Extended dataset:

```bash
python train.py --checkpoint_dir /CHECKPOINT_PATH/ --transformer_ver cat --batch_size 32 --epochs 80 --savelog /SAVELOG_PATH/ --detloss giou --claloss focal --uncer True
```

---

## Evaluation

To evaluate on EndoVis17/18-VQLA-Extended:

```bash
python train.py --validate True --checkpoint_dir /CHECKPOINT_PATH/ --transformer_ver cat --batch_size 32
```

## Checkpoint

Here is the checkpoint: [Checkpoint](https://drive.google.com/file/d/1n37lHo4xLYC-bbCpe-511x5Jsm5pxa5G/view?usp=sharing).

---
