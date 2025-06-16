from pathlib import Path


MODEL_WEIGHTS = "blink_best.pth"
STATS_NPZ = "blink_stats.npz"
BLINKING_THREASHOLD = 0.50

class Paths:
    ROOT_DIR      = Path(__file__).parent
    DATA_DIR      = ROOT_DIR / "data"
    RAW_DATA_DIR  = DATA_DIR / "raw_sessions"
    DATASET_IMG   = DATA_DIR / "blink_dataset_img.csv"
    DATASET_NUM   = DATA_DIR / "blink_dataset_num.csv"
    MODEL_DIR     = ROOT_DIR / "dev" / "models"
    IMG_WEIGHTS   = MODEL_DIR / "blink_img_best.pth"
    NUM_WEIGHTS   = MODEL_DIR / "blink_num_best.pth"
    STATS_DIR     = ROOT_DIR / "dev" / "stats"
    IMG_STATS_NPZ = STATS_DIR / "blink_img_stats.npz"
    NUM_STATS_NPZ = STATS_DIR / "blink_num_stats.npz"


# Images:
class Image_Constants:

    IM_WIDTH = 24
    IM_HEIGHT = 12

    LEFT_EYE_OUT_ID, LEFT_EYE_INSIDE_ID, LEFT_EYE_UP_ID, LEFT_EYE_LOW_ID = (
        33,
        133,
        159,
        145,
    )
    RIGHT_EYE_OUT_ID, RIGHT_EYE_INSIDE_ID, RIGHT_EYE_UP_ID, RIGHT_EYE_LOW_ID = (
        362,
        263,
        386,
        374,
    )

    LEFT_EYE_IDS = (
        33, 133, 144, 145, 153, 154, 155, 157, 158, 159, 160, 161, 163, 173
    )
    RIGHT_EYE_IDS = (
        263, 362, 373, 374, 380, 381, 382, 384, 385, 386, 387, 388, 390, 398
    )

    LEFT_EYE_PAIR_IDS = (
        (144, 158),
        (145, 159),
        (153, 160),
        (154, 161),
        (155, 163),
        (157, 173),
    )

    RIGHT_EYE_PAIR_IDS = (
        (373, 385),
        (374, 386),
        (380, 387),
        (381, 388),
        (382, 390),
        (384, 398),
    )

    ID_ARRAYS = LEFT_EYE_IDS + RIGHT_EYE_IDS


# Blinking frames
class Data_Gathering_Constants:
    BLINK_PRE_FRAMES = 3
    BLINK_POST_FRAMES = 6

    NUM_COLS = [
        "ratio_left",
        "ratio_right",
        "v1_left",
        "v2_left",
        "v3_left",
        "v4_left",
        "v5_left",
        "v6_left",
        "v1_right",
        "v2_right",
        "v3_right",
        "v4_right",
        "v5_right",
        "v6_right",
        "width_left",
        "width_right",
    ]


# Constants for different models
class Model_Constants:
    NUM_FEATURES = 16

    class RATIO_MODEL_CONSTANTS:
        FC_SIZES = (64, 64, 32)
        LTSM_INPUT_SIZE = FC_SIZES[-1]
        LTSM_HIDDEN = 32
        LTSM_LAYERS = 1
        BIDIRECTIONAL = True


class Training_Constnats:
    SEQUENCE_LENGTH = 30
    SPLIT_RATIO = 0.6
    # Path to the training CSV inside the repository
    CSV_PATH = str(Paths.ROOT_DIR / "dev" / "blinkdata.csv")
    BATCH_SIZE = 32
    CURRENT_BEST_F1 = 0.411
