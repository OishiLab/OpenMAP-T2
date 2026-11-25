from __future__ import annotations

import os

import numpy as np
import pandas as pd

from utils.output.make_csv import LEVEL_DIR
from utils.output.output_space import save_label_volume

LEVELS = (
    "Type1_Level1",
    "Type1_Level2",
    "Type1_Level3",
    "Type1_Level4",
    "Type2_Level1",
    "Type2_Level2",
    "Type2_Level3",
    "Type2_Level4",
    "Type2_Level5",
)


def create_parcellated_images(output, output_dir, basename, odata, data, output_ext=".nii.gz", output_space="native"):
    """Write hierarchical label maps from the Type1_Level5 volume."""
    df_no = pd.read_csv(os.path.join(LEVEL_DIR, "Level_ROI_No.csv"))

    # Remap each level through a lookup table instead of one full-volume pass per label.
    # Replacements are applied sequentially on the value domain, which matches the original
    # in-place loop (including chained replacements).
    domain = np.arange(int(output.max()) + 1, dtype=np.int64)

    for level in LEVELS:
        lut = domain.copy()
        for old, new in zip(df_no["Type1_Level5"], df_no[level]):
            lut[lut == old] = new
        label = lut.astype(output.dtype)[output]
        save_label_volume(output_dir, "parcellated", basename, level, label, odata, data, output_ext, output_space)
