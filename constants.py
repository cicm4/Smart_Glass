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

    ID_ARRAYS = (
        LEFT_EYE_OUT_ID,
        LEFT_EYE_INSIDE_ID,
        LEFT_EYE_UP_ID,
        LEFT_EYE_LOW_ID,
        RIGHT_EYE_OUT_ID,
        RIGHT_EYE_INSIDE_ID,
        RIGHT_EYE_UP_ID,
        RIGHT_EYE_LOW_ID,
    )


# Blinking frames
class Data_Gathering_Constants:
    BLINK_PRE_FRAMES = 3
    BLINK_POST_FRAMES = 6

    NUM_COLS = [
        "ratio_left",
        "ratio_right",
    ]


# Constants for different models
class Model_Constants:
    NUM_FEATURES = 2

    class MEDIUM_MODEL_CONSTANTS:
        CONVOLUTIONAL_IMAGE_CHANELS = (16, 64, 32)
        CONVOLUTIONAL_NUMERIC_CHANELS = (64, 16)
        IMAGE_OUTPUT_SIZE = 64
        LTSM_INPUT_SIZE = IMAGE_OUTPUT_SIZE + CONVOLUTIONAL_NUMERIC_CHANELS[-1]
        LTSM_HIDDEN = 64
        LTSM_LAYERS = 3
        BIDIRECTIONAL = True

    class SMALL_MODEL_CONSTANTS:
        CONVOLUTIONAL_IMAGE_CHANELS = (16, 32)
        CONVOLUTIONAL_NUMERIC_CHANELS = (32, 16)
        IMAGE_OUTPUT_SIZE = 64
        LTSM_INPUT_SIZE = IMAGE_OUTPUT_SIZE + CONVOLUTIONAL_NUMERIC_CHANELS[-1]
        LTSM_HIDDEN = 64
        LTSM_LAYERS = 1
        BIDIRECTIONAL = True

    class XS_MODEL_CONSTANTS:
        CONVOLUTIONAL_IMAGE_CHANELS = (16, 24)
        CONVOLUTIONAL_NUMERIC_CHANELS = (24, 16)
        IMAGE_OUTPUT_SIZE = 64
        LTSM_INPUT_SIZE = IMAGE_OUTPUT_SIZE + CONVOLUTIONAL_NUMERIC_CHANELS[-1]
        LTSM_HIDDEN = 32
        LTSM_LAYERS = 1
        BIDIRECTIONAL = False

    class RATIO_MODEL_CONSTANTS:
        FC_SIZES = (64, 64, 32)
        LTSM_INPUT_SIZE = FC_SIZES[-1]
        LTSM_HIDDEN = 64
        LTSM_LAYERS = 2
        BIDIRECTIONAL = True


class Training_Constnats:
    SEQUENCE_LENGTH = 30
    SPLIT_RATIO = 0.6
    # Path to the training CSV inside the repository
    CSV_PATH = str(Paths.ROOT_DIR / "dev" / "blinkdata.csv")
    BATCH_SIZE = 32
    CURRENT_BEST_F1 = 0.01
