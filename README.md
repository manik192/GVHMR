# GVHMR — Google Colab Reproduction

**World-Grounded Human Motion Recovery via Gravity-View Coordinates**  

This notebook reproduces the GVHMR pipeline on Google Colab with a free T4 GPU. It includes environment setup, benchmark evaluation on 3DPW / RICH / EMDB, speed optimizations, and a Gradio demo interface.

---

## Requirements

- Google account (for Colab + Google Drive)
- Google Colab with **T4 GPU** (free tier works)
- ~5 GB free space on Google Drive

---

## One-Time Setup (Do This First, Only Once Ever)

### 1. Body Models (Manual Download — Required)

SMPL and SMPLX require registration on their official websites. Do this once, save to Drive.

**SMPL:**
1. Go to https://smpl.is.tue.mpg.de → Register → Download → `SMPL_python_v.1.1.0.zip`
2. Extract and find `SMPL_NEUTRAL.pkl`
3. Upload to `MyDrive/smpl/SMPL_NEUTRAL.pkl`

**SMPLX:**
1. Go to https://smpl-x.is.tue.mpg.de → Register → Download → `models_smplx_v1_1.zip`
2. Extract and find `SMPLX_NEUTRAL.npz`
3. Upload to `MyDrive/smplx/SMPLX_NEUTRAL.npz`

> If you skip this, the notebook falls back to a Hugging Face mirror automatically — but the official files are more reliable.

### 2. Build pytorch3d (One-Time, ~35 Minutes)

There is no prebuilt wheel for Python 3.12. You need to build it once and save to Drive.

In the notebook, find the cell marked **"FIRST-TIME BUILD"** and uncomment it. Run it once. It saves the compiled package to `MyDrive/pytorch3d_built/`. All future sessions just copy from Drive — takes 10 seconds.

---

## Running the Notebook

### Step 1 — Open in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manik192/GVHMR/blob/main/GVHMR.ipynb)

Or go to [colab.research.google.com](https://colab.research.google.com) → File → Open notebook → GitHub → paste this repo URL.

### Step 2 — Set Runtime to GPU

**Do this before running any cell.** The notebook will not work on CPU.

### Step 3 — Run Cells in Order

Run each cell top to bottom. Do not skip cells. Each section depends on the previous one.

| Cell | What it does | Time |
|---|---|---|
| Clone + Install | Clones GVHMR repo, installs Python packages | ~3 min |
| pytorch3d | Loads from Drive (or builds first time) | 10 sec / 35 min |
| Patches | Fixes Python 3.12 + PyTorch 2.6 compatibility | ~10 sec |
| GPU check | Confirms CUDA is active | ~5 sec |
| Checkpoints | Downloads model weights from Hugging Face | ~5 min |
| Demo | Runs inference on the bundled tennis video | ~108 sec |
| Benchmark Eval | Runs 3DPW + RICH + EMDB evaluation | ~20 min |
| Gradio | Launches a web interface to upload your own video | instant |

---

## Running Your Own Video

### Option A — Direct Cell

```python
!python /content/GVHMR/tools/demo/demo.py \
    --video=/path/to/your/video.mp4 \
    -s   # remove -s if camera is moving (handheld)
```

### Option B — Gradio Interface

Run the Gradio cell at the bottom of the notebook. A public link will appear — open it on any device including your phone. Upload a video, click **Run GVHMR**, wait ~108 seconds, get the result.

**Flags:**
- `-s` — static camera (tripod shot). Use this for most videos. Skips visual odometry.
- No flag — moving camera (handheld). Runs SimpleVO for camera motion estimation, slightly slower.

---

## Speed Optimizations Applied

The notebook includes 5 optimizations over the original GVHMR codebase:

| Change | Effect |
|---|---|
| ViTPose batch 16 → 32 | 44s → 23s |
| flip_test disabled | Halves ViTPose iterations |
| HMR2 batch 16 → 32 | 21s → 13s |
| YOLO imgsz=640, half=True | 26s → 16s |
| CRF 23 → 28 | Faster video encoding |
| **Total** | **145s → 108s (−26%)** |

---

## Benchmark Evaluation

To reproduce the paper's results, you need the eval datasets. Run the dataset download cell first:

```python
# Uncomment in notebook:
# !gdown --folder https://drive.google.com/drive/folders/1bWQceGEpXrqE8OXYpuQPUsRHANdxI_9q -O inputs/
```

Then run the benchmark cell:

```python
!python tools/train.py \
    global/task=gvhmr/test_3dpw_emdb_rich \
    exp=gvhmr/mixed/mixed \
    ckpt_path=inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt
```

**Our reproduced results vs paper:**

| Dataset | Metric | Paper | Ours | Diff |
|---|---|---|---|---|
| RICH | WA-MPJPE | 78.8 | 78.8 | 0.0 ✅ |
| RICH | RTE | 2.4 | 2.4 | 0.0 ✅ |
| EMDB | RTE | 1.9 | 1.9 | 0.0 ✅ |
| EMDB | Jitter | 16.5 | 16.1 | −0.4 ✅ |
| 3DPW | PA-MPJPE | 36.2 | 37.5 | +1.3 |

---

## Common Errors and Fixes

**`pickle data was truncated`**  
Your SMPL_NEUTRAL.pkl downloaded corrupted. Delete it and re-download:
```python
!rm inputs/checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
!aria2c -c -x 8 https://huggingface.co/camenduru/SMPLer-X/resolve/main/SMPL_NEUTRAL.pkl \
    -d inputs/checkpoints/body_models/smpl -o SMPL_NEUTRAL.pkl
```

**`FileNotFoundError: pytorch3d`**  
Your Drive cache is missing or the copy failed. Re-run the pytorch3d cell. If this is your first time, uncomment the build block.

**`AssertionError: Video not found`**  
The video path does not exist. Make sure you are passing the full absolute path starting with `/content/`.

**`weights_only` errors on torch.load**  
The patches cell did not run. Re-run the patches cell before running the demo.

**`0% GPU utilization` in Colab dashboard**  
This is a display lag — the GPU is being used. Confirm by checking the log for `[GPU]: Tesla T4`. If it says `device: cpu`, the runtime was not set to GPU — restart and change runtime type first.

**Gradio connection drops / OOM crash**  
The Gradio cell uses subprocess so each run loads models fresh. If it crashes, reduce your input video to under 15 seconds or use the direct `!python` cell instead.

---

