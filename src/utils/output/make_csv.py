from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
import pandas as pd

LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "level"))

# Frontal, central, and parietal sulci (external/internal), excluding the Sylvian fissure.
SULCUS_LABELS = (250, 251, 252, 253, 254, 255, 256, 257, 260, 261, 262, 263)
SYLVIAN_LABELS = (258, 259)

N_REGIONS = 280
LEVELS = (
    "Type1_Level5",
    "Type1_Level4",
    "Type1_Level3",
    "Type1_Level2",
    "Type1_Level1",
    "Type2_Level5",
    "Type2_Level4",
    "Type2_Level3",
    "Type2_Level2",
    "Type2_Level1",
)


def change_level(df, level="Type1_Level1", sulcus=True):
    """Aggregate Type1_Level5 region columns into a coarser level."""
    ROI_number = pd.read_csv(os.path.join(LEVEL_DIR, "Level_ROI_No.csv"))
    ROI_name = pd.read_csv(os.path.join(LEVEL_DIR, "Level_ROI_Name.csv"))

    if sulcus is False:
        tmp = ROI_number["Type1_Level2"]
        keep = (tmp != 18) & (tmp != 19)
        ROI_number = ROI_number[keep]
        ROI_name = ROI_name[keep]
    data = dict(zip(ROI_number["ROI"], ROI_number[level]))
    level_dict = defaultdict(list)
    for key, value in data.items():
        level_dict[str(value)].append(key)

    # Name each output column from the regions that share its number, never by position.
    names_by_number = defaultdict(list)
    for number, name in zip(ROI_number[level], ROI_name[level]):
        if name not in names_by_number[str(number)]:
            names_by_number[str(number)].append(name)
    conflicts = {n: names for n, names in names_by_number.items() if len(names) > 1}
    if conflicts:
        raise ValueError(f"{level}: regions with the same number have different names in Level_ROI_Name.csv: {conflicts}")

    change_df_list = [df[members].sum(axis=1).rename(names_by_number[number][0]) for number, members in level_dict.items()]
    return pd.concat(change_df_list, axis=1)


def _write_sylvian_ratio(counts: np.ndarray, output_dir: str, basename: str) -> None:
    sum_sulcus = int(sum(counts[label] for label in SULCUS_LABELS))
    sum_syl = int(sum(counts[label] for label in SYLVIAN_LABELS))
    ratio = sum_syl / sum_sulcus if sum_sulcus else None
    table = pd.DataFrame(
        {
            "SylvianRatio": [ratio],
            "SylvianFissure_L+R": [sum_syl],
            "Sulcus_L+R": [sum_sulcus],
            "SylvianRatio_Caluclation_Formula": ["(Sylvian Fissure L+R)/(Sulcus L+R)"],
            "Sulcus_L+R_Caluclation_Formula": ["(Frontal Sulcus LR)+(Central Sulcus LR)+(Parietal Sulcus LR)"],
        }
    )
    table.to_csv(os.path.join(output_dir, "csv", f"{basename}_SylvianRatio.csv"), index=False)


def make_csv(parcellation, output_dir, basename):
    """Write regional volumes (mm³ on the 1 mm grid) for every level, and the Sylvian fissure ratio."""
    names = pd.read_table(os.path.join(LEVEL_DIR, "Type1Level5.txt"), names=["number", "region"])
    counts = np.bincount(parcellation.ravel(), minlength=N_REGIONS + 1)
    volumes = counts[names["number"].astype(int).to_numpy()]
    df_type1_level5 = pd.DataFrame([volumes], columns=names["region"].to_numpy())

    os.makedirs(os.path.join(output_dir, "csv"), exist_ok=True)
    for level in LEVELS:
        frame = df_type1_level5 if level == "Type1_Level5" else change_level(df_type1_level5, level=level)
        frame.to_csv(os.path.join(output_dir, "csv", f"{basename}_{level}.csv"), index=False)
    _write_sylvian_ratio(counts, output_dir, basename)
    return df_type1_level5
