import argparse
import glob
import os
import sys
from functools import partial

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm as std_tqdm

tqdm = partial(std_tqdm, dynamic_ncols=True)

from utils.models.load_model import load_model
from utils.output.make_csv import make_csv
from utils.output.make_level import create_parcellated_images
from utils.output.output_space import OUTPUT_SPACE_CHOICES, save_label_volume
from utils.output.processing_log import ProcessingLog
from utils.pipeline.hemisphere import hemisphere
from utils.pipeline.parcellate import parcellation
from utils.pipeline.postprocessing import postprocessing
from utils.pipeline.preprocessing import preprocessing
from utils.pipeline.stripping import stripping


def create_parser():
    parser = argparse.ArgumentParser(description="Run inference with OpenMAP-T2 on T2-weighted brain MRI.")
    parser.add_argument("-i", required=True, help="Input folder containing NIfTI files (.nii or .nii.gz).")
    parser.add_argument("-o", required=True, help="Output folder. Created if it does not exist.")
    parser.add_argument("-m", required=True, help="Folder of pretrained SSNet, PNet, and HNet weights.")
    parser.add_argument(
        "--output-ext",
        default=".nii.gz",
        choices=[".nii.gz", ".nii"],
        help="Extension for saved NIfTI files (default: .nii.gz).",
    )
    parser.add_argument(
        "--output-space",
        default="native",
        choices=list(OUTPUT_SPACE_CHOICES),
        help=(
            "Geometry for saved NIfTI files. "
            "'native': resample to the N4-corrected canonical input grid (default). "
            "'conform': keep the 1 mm isotropic 256³ grid. "
            "'both': write both (1 mm files under conform/)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Slices per forward pass (default: 16).")
    parser.add_argument(
        "--only-skull-stripping",
        action="store_true",
        help="Stop after skull stripping. Parcellation is skipped.",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    print("Parsed arguments:", args)
    return args


def _basename(path: str) -> str:
    basename = os.path.splitext(os.path.basename(path))[0]
    if basename.endswith(".nii"):
        basename = os.path.splitext(basename)[0]
    return basename


def main():
    print(
        "\n#######################################################################\n"
        "Please cite the following link when using OpenMAP-T2:\n"
        "https://github.com/OishiLab/OpenMAP-T2 \n"
        "#######################################################################\n"
    )
    opt = create_parser()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    use_amp = (not opt.no_amp) and device.type == "cuda"
    print(f"Using device: {device} (amp={use_amp})")

    try:
        ssnet, pnet, hnet = load_model(opt, device)
        print("Load complete !!")
    except Exception as exc:
        print(f"Error during model loading: {exc}")
        sys.exit(1)

    if not os.path.isdir(opt.i):
        print(f"Error: Input directory {opt.i} does not exist.")
        sys.exit(1)

    pathes = sorted(
        glob.glob(os.path.join(opt.i, "**/*.nii"), recursive=True)
        + glob.glob(os.path.join(opt.i, "**/*.nii.gz"), recursive=True)
    )
    print(f"Found {len(pathes)} NIfTI files in {opt.i}")

    processing_log = ProcessingLog()
    os.makedirs(opt.o, exist_ok=True)

    for path in tqdm(pathes):
        basename = None
        try:
            basename = _basename(path)
            output_dir = os.path.join(opt.o, basename)
            os.makedirs(os.path.join(output_dir, "original"), exist_ok=True)

            original = nib.squeeze_image(nib.as_closest_canonical(nib.load(path)))
            nib.save(
                nib.Nifti1Image(original.get_fdata().astype(np.float32), affine=original.affine),
                os.path.join(output_dir, f"original/{basename}{opt.output_ext}"),
            )

            odata, data = preprocessing(path, output_dir, basename, opt.output_ext)
            cropped, shift = stripping(
                output_dir,
                basename,
                odata,
                data,
                ssnet,
                device,
                opt.output_ext,
                opt.output_space,
                opt.batch_size,
                use_amp,
            )
            if opt.only_skull_stripping:
                processing_log.add_skipped(path, basename, "Pipeline stopped early (--only-skull-stripping).")
                continue

            parcellated = parcellation(cropped, pnet, device, opt.batch_size, use_amp)
            separated = hemisphere(cropped, hnet, device, opt.batch_size, use_amp)
            output = postprocessing(parcellated, separated, shift)
            processing_log.check_low_volume(path, basename, output)
            make_csv(output, output_dir, basename)
            save_label_volume(
                output_dir,
                "parcellated",
                basename,
                "Type1_Level5",
                output,
                odata,
                data,
                opt.output_ext,
                opt.output_space,
            )
            create_parcellated_images(output, output_dir, basename, odata, data, opt.output_ext, opt.output_space)
            del odata, data, cropped, output
        except Exception as exc:
            print(f"Error processing {path}: {exc}")
            processing_log.add_failed(path, basename or os.path.basename(path), str(exc))
            continue

    processing_log.save(opt.o)


if __name__ == "__main__":
    main()
