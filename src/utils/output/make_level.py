import os

import numpy as np
import pandas as pd

from utils.output.output_space import save_label_volume

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
LEVEL_DIR = os.path.join(PROJECT_ROOT, "level")


def create_parcellated_images(output, output_dir, basename, odata, data, output_ext=".nii.gz", output_space="native"):
    """Write hierarchical label maps from the Type1 Level5 volume."""
    df_no = pd.read_csv(os.path.join(LEVEL_DIR, "Level_ROI_No.csv"))

    all_level = [
        "Type1_Level1",
        "Type1_Level2",
        "Type1_Level3",
        "Type1_Level4",
        "Type2_Level1",
        "Type2_Level2",
        "Type2_Level3",
        "Type2_Level4",
        "Type2_Level5",
    ]

    for level in all_level:
        mapping = dict(zip(df_no["Type1_Level5"], df_no[level]))
        label = np.copy(output)
        for old, new in mapping.items():
            label[label == old] = new
        save_label_volume(
            output_dir,
            "parcellated",
            basename,
            level,
            label,
            odata,
            data,
            output_ext,
            output_space,
        )
