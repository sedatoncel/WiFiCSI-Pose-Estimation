# WiFiCSI-Pose-Estimation
# 3D Human Pose Estimation from WiFi CSI Signals

A Transformer-based model that estimates a 3D human skeleton (17 joints) from WiFi
Channel State Information (CSI), trained and evaluated on the **MMFi** dataset.
The project focuses on **cross-environment generalization**: how well a model trained
in some rooms works in rooms it has never seen.

## Summary of results

| Setting | MPJPE | Y (depth) MAE | X / Z MAE |
|---|---|---|---|
| Train (E01+E02, seen rooms) | **76.1 mm** | 39.2 mm | 36.2 / 31.3 mm |
| Validation (E04, unseen) | 133.9 mm | 79.5 mm | 58.9 / 51.7 mm |
| Test (E03, unseen, unbiased) | 137.4 mm | 85.9 mm | 57.3 / 51.1 mm |
| Baseline (always predict mean pose) | ~135 mm | 75.1 mm | 54.4 / 44.3 mm |

In the rooms used for training, the model clearly beats a "mean pose" baseline,
including on the depth axis. In unseen rooms it drops to the baseline level. Two
different unseen rooms give almost the same error, which shows this is a consistent
limitation, not a problem with one single room.

## Model

`CSITransformerEncoder` (in `file3_model.py`):

* Input: `(B, 114, 30)` — 114 subcarrier tokens, each of dimension 30 (3 antennas x 10 packets, amplitude).
* Linear projection to `d_model = 128`, a learnable CLS token, and sinusoidal positional encoding.
* 4 Transformer encoder layers, 8 attention heads, Pre-LayerNorm, GELU.
* Two output heads: root position (3 values) and 17 keypoints (51 values).
* ~14.4M parameters.

![Model pipeline](model_diagram.png)

## Files

| File | Description |
| `file3_model.py` | Model definition (CSITransformerEncoder). |
| `egitim_validation.py` | Training + validation + generates the result plots. |
| `e03_test_gorsel.py` | Loads the trained model, tests on E03, draws real-vs-predicted skeletons and saves an animation. |
| `mmfi.py` | Official MMFi dataset library. |
| `config.yaml` | Dataset / split configuration (cross-scene, WiFi-CSI). |


## Dataset

This project uses the WiFi-CSI modality of the **MMFi** dataset. The data is **not**
included in this repository (it is very large). Download it from the official source:

* MMFi paper: Yang et al., *MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for
  Versatile Wireless Sensing*, NeurIPS 2023 (arXiv:2305.10345).
* Dataset download: https://drive.google.com/drive/folders/1tf9HHfCGo2lsw_km5R481emmdBmpuyJy

After downloading, set the dataset path inside the scripts:

```python
dataset_root = r"C:\path\to\MMFi_Dataset"
```

(change this line in `egitim_validation.py`, `e03_test_gorsel.py` and `grafik_duzelt.py`).

## Data split (cross-scene protocol)

* **Train:** rooms E01 + E02
* **Validation:** room E04 (checked every epoch)
* **Test:** room E03 (held out, used only once for the final unbiased result)

The split is set through `config.yaml`. Each room is a physically different
environment, so testing on a new room measures real generalization.

## Setup

```bash
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended. The model was trained on a single NVIDIA RTX 3060.

## How to run

**1. Train the model** (also produces the plots in `./grafikler/`):

```bash
python egitim_validation.py
```

This trains for 30 epochs and saves `wifi_iskelet_modeli_son.pth` (last epoch) and
`wifi_iskelet_modeli_best.pth` (best validation epoch).

**2. Test on the unseen room E03 and create visualizations:**

```bash
python e03_test_gorsel.py
```

This prints the E03 test metrics, opens real-vs-predicted skeleton plots, and saves an
animation (`e03_animasyon.gif` / `.mp4`).

> If the plot windows do not open (you see `FigureCanvasAgg is non-interactive`), add
> `import matplotlib; matplotlib.use("TkAgg")` at the top of the script, or run
> `%matplotlib qt` in the console before running.

## Training details

* Optimizer: Adam, learning rate 1e-4
* Scheduler: CosineAnnealingLR (30 epochs)
* Loss: L1 on keypoints + 0.1 x L1 on root
* Batch size: 32
* Root-relative alignment (pelvis = origin)
* Raw CSI amplitude, no filtering, trained end to end

## Evaluation

* **MPJPE** — mean per-joint position error (mm).
* **Per-axis MAE** — error on X, Y, Z separately (Y is the depth axis and the hardest).
* **PCK** — percentage of joints within a distance threshold.
* **Mean-pose baseline** — a sanity check the model must beat.


## Limitations and future work

* The model overfits to the training rooms and does not transfer well to new rooms.
* Depth (Y) is the weakest axis, due to WiFi's low front-to-back resolution.
* **Future work:** use the CSI **phase** (not only amplitude), apply **filtering /
  phase-sanitization** methods (e.g. a Hampel filter for outliers and a smoothing or
  linear-fit step), train on more and more varied rooms, and add domain adaptation and
  temporal (multi-frame) modeling.

## Acknowledgements

MMFi dataset: Yang, J., Huang, H., Zhou, Y., Chen, X., Xu, Y., Yuan, S., Zou, H.,
Lu, C. X., Xie, L. *MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for Versatile
Wireless Sensing.* NeurIPS 2023, Datasets and Benchmarks Track.
