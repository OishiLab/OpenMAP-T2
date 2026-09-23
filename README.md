# OpenMAP-T2

**OpenMAP-T2 parcellates the whole brain into 280 anatomical regions from a T2-weighted MRI in about 70 seconds per case.**

## Requirements

- **Python 3.11 or later** (required by pandas 3.0)
- **PyTorch 2.12 or later** (installed automatically with `uv sync`; install separately when using pip)
- Other pinned libraries: nibabel 5.4.2, pandas 3.0.3, scipy 1.17.1, scikit-image 0.25.2, SimpleITK 2.5.5, tqdm 4.68.2

## Installation with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver written in Rust. It provides a faster alternative to pip for managing dependencies.

0. Install uv

   **macOS and Linux:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   **Windows:**
   ```powershell
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

   Or using pip:
   ```bash
   pip install uv
   ```

1. Clone this repository, and go into the repository:
```bash
git clone https://github.com/OishiLab/OpenMAP-T2.git
cd OpenMAP-T2
```

2. Install dependencies with uv (including PyTorch):
```bash
uv sync
```

3. Download the pretrained weights. Until peer review is complete, they are released publicly on Google Drive (no application is required):

[OpenMAP-T2 pretrained weights](https://drive.google.com/drive/folders/1SUirFIbWI8h0wWLzDcIhvogIRImz76GK?usp=sharing)

Download that folder and pass it as `MODEL_FOLDER`. It is laid out as follows:

```
MODEL_FOLDER/
  ├ SSNet/SSNet.pth
  ├ PNet/PNet.pth
  └ HNet/HNet.pth
```

SSNet, PNet, and HNet are each a single 2.5D U-Net. PNet and HNet read the slicing plane from an extra input channel, so there is one weight file per network rather than one file per anatomical view.

A training checkpoint directory is also accepted. If `SSNet/SSNet.pth` is absent, OpenMAP-T2 loads `stripping/models/model.pth`, `parcellation/models/model.pth`, and `hemisphere/models/model.pth` (or the latest `epoch_*.pth`).

## Installation with pip (Alternative)

0. Install Python 3.11 or later and create a virtual environment.

1. Clone this repository, and go into the repository:
```bash
git clone https://github.com/OishiLab/OpenMAP-T2.git
cd OpenMAP-T2
```

2. Install PyTorch compatible with your environment.
   <https://pytorch.org/>

   The latest stable PyTorch (2.12+) requires **Python 3.10 or later**; this project requires **Python 3.11 or later** because of pandas 3.0.

   If you want to install an older PyTorch environment, you can download it from <https://pytorch.org/get-started/previous-versions/>.

3. Install libraries other than PyTorch:
```bash
pip install -r requirements.txt
```

4. Download the pretrained weights from the Google Drive folder above and place them in `MODEL_FOLDER`.

## How to run

GPU execution is selected automatically when CUDA or Apple MPS is available.

```bash
# uv (if you installed with uv sync)
uv run python src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```

```bash
# pip (activate the virtual environment first: source .venv/bin/activate)
python3 src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```

To pin a device:

```bash
# uv
CUDA_VISIBLE_DEVICES=1 uv run python src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```

```bash
# pip
CUDA_VISIBLE_DEVICES=1 python3 src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```

On Windows, use `python` instead of `python3`.

### Options

* **`-i`**: Folder of input NIfTI files (`.nii` or `.nii.gz`). Subfolders are included.
* **`-o`**: Output folder. Created if missing.
* **`-m`**: Pretrained-model folder.
* **`--output-ext {.nii.gz,.nii}`**: Extension of saved images. Default is `.nii.gz`.
* **`--output-space {native,conform,both}`**: Grid of saved images. Default is `native`.
  * `native`: resample back to the N4-corrected canonical input geometry.
  * `conform`: keep the internal 1 mm isotropic 256³ grid. Filenames include `_1mm`.
  * `both`: write both. The 1 mm copies are under `conform/`.
* **`--only-skull-stripping`**: Stop after brain extraction.
* **`--batch-size`**: Slices per forward pass. Default is 16.
* **`--no-amp`**: Run CUDA inference in full precision.

```bash
# uv
uv run python src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --output-space conform
uv run python src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --only-skull-stripping
```

```bash
# pip
python3 src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --output-space conform
python3 src/parcellation.py -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER --only-skull-stripping
```

CSV regional volumes are always counted on the 1 mm grid. Each voxel is 1 mm³, so the values are whole millimeters cubed.

## What the pipeline does

1. N4 bias-field correction, canonical orientation, and resampling to 1 mm 256³ when needed.
2. Skull stripping with a 3-channel 2.5D network on axial, coronal, and sagittal views.
3. Histogram matching of the stripped brain, then a center-of-mass crop to 224³.
4. 139-class parcellation (axial, coronal, sagittal) and hemisphere segmentation (axial, coronal).
5. Fusion to 280 regions, including the internal/external split of selected sulcal CSF, then return to the 256³ grid.

## Output

```
INPUT_FOLDER/
  ├ A.nii.gz
  └ B.nii.gz

OUTPUT_FOLDER/
  ├── failed_cases.csv          # only when a case fails, is skipped, or is implausibly small
  └── A/
      ├── original/
      │   ├── A.nii.gz
      │   └── A_N4.nii.gz
      ├── stripped/
      │   ├── A_stripped.nii.gz
      │   └── A_stripped_mask.nii.gz
      ├── parcellated/
      │   ├── A_Type1_Level1.nii.gz
      │   ├── ...
      │   └── A_Type1_Level5.nii.gz
      └── csv/
          ├── A_Type1_Level1.csv
          ├── ...
          ├── A_Type2_Level5.csv
          └── A_SylvianRatio.csv
```

`failed_cases.csv` lists `case_id`, `input_path`, `status` (`failed`, `skipped`, or `low_volume`), `reason`, and `total_brain_volume_mm3`. A case is `low_volume` when the 1 mm labeled brain is under 10,000 mm³. `--only-skull-stripping` records the case as `skipped`.

With `--output-space conform`, the same folders are used and filenames include `_1mm`. With `--output-space both`, those 1 mm files are written under `conform/`.

## Level metadata

`level/` holds the JHU-atlas lookup tables.

* `level/OpenMAP-T2_multilevel_lookup_table_dictionary.csv` maps Type1 Level5 names onto the coarser levels.
* `level/Level1.txt` through `level/Level5.txt` are the Type1 name lists.

## FAQ

* **How much GPU memory do I need?**
  Experiments used an NVIDIA RTX 3090 (24 GB). A typical volume uses well under that, but memory scales with the image because inference is fully convolutional. If you run out of memory, confirm the voxel size, keep the field of view on the head, and lower `--batch-size`.
* **Will you provide the training code?**
  No. Training is tied to data that cannot be released.

## Docker

```bash
docker build -t openmap-t2 .
docker run --rm -it -v "$(pwd):/app" openmap-t2 -i INPUT_FOLDER -o OUTPUT_FOLDER -m MODEL_FOLDER
```

The image installs a CPU build of PyTorch. For a GPU, prefer a local install and `CUDA_VISIBLE_DEVICES`.
