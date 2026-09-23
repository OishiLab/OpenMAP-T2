import os
from collections import defaultdict

import numpy as np
import pandas as pd

LEVEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "level"))

# Frontal, central, and parietal sulci (external/internal), excluding the Sylvian fissure.
SULCUS_LABELS = (250, 251, 252, 253, 254, 255, 256, 257, 260, 261, 262, 263)
SYLVIAN_LABELS = (258, 259)


def change_level(df, level="Type1_Level1", sulcus=True):
    """Aggregate Type1 Level5 region columns into a coarser level."""
    ROI_number = pd.read_csv(os.path.join(LEVEL_DIR, "Level_ROI_No.csv"))
    ROI_name = pd.read_csv(os.path.join(LEVEL_DIR, "Level_ROI_Name.csv"))

    if sulcus is False:
        tmp = ROI_number["Type1_Level2"]
        ROI_number = ROI_number[tmp != 18]
        ROI_number = ROI_number[tmp != 19]
        ROI_name = ROI_name[tmp != 18]
        ROI_name = ROI_name[tmp != 19]
    data = dict(zip(ROI_number["ROI"], ROI_number[level]))
    level_dict = defaultdict(list)
    for key, value in data.items():
        level_dict[str(value)].append(key)

    change_df_list = []
    for i, (_key, value) in enumerate(level_dict.items()):
        name = ROI_name[level].unique()[i]
        change_df_list.append(df[value].sum(axis=1).rename(name))

    return pd.concat(change_df_list, axis=1)


def _write_sylvian_ratio(parcellation, output_dir, basename):
    sum_sulcus = int(sum(np.count_nonzero(parcellation == label) for label in SULCUS_LABELS))
    sum_syl = int(sum(np.count_nonzero(parcellation == label) for label in SYLVIAN_LABELS))
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
    """Write regional volumes (mm³ on the 1 mm grid) and the Sylvian fissure ratio."""
    csv_path = os.path.join(LEVEL_DIR, "Level5.txt")
    df_type1_level5 = pd.read_table(csv_path, names=["number", "region"]).astype("str").set_index("number")
    for i in range(1, 281):
        df_type1_level5.loc[str(i), basename] = np.count_nonzero(parcellation == i)

    df_type1_level5 = df_type1_level5.set_index("region").T.reset_index(drop=True)
    levels = {
        "Type1_Level5": df_type1_level5,
        "Type1_Level4": change_level(df_type1_level5, level="Type1_Level4"),
        "Type1_Level3": change_level(df_type1_level5, level="Type1_Level3"),
        "Type1_Level2": change_level(df_type1_level5, level="Type1_Level2"),
        "Type1_Level1": change_level(df_type1_level5, level="Type1_Level1"),
        "Type2_Level5": change_level(df_type1_level5, level="Type2_Level5"),
        "Type2_Level4": change_level(df_type1_level5, level="Type2_Level4"),
        "Type2_Level3": change_level(df_type1_level5, level="Type2_Level3"),
        "Type2_Level2": change_level(df_type1_level5, level="Type2_Level2"),
        "Type2_Level1": change_level(df_type1_level5, level="Type2_Level1"),
    }

    os.makedirs(os.path.join(output_dir, "csv"), exist_ok=True)
    for level, frame in levels.items():
        frame.to_csv(os.path.join(output_dir, "csv", f"{basename}_{level}.csv"), index=False)
    _write_sylvian_ratio(parcellation, output_dir, basename)
    return df_type1_level5
