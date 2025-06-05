MODEL_WEIGHTS = "blink_best.pth"
STATS_NPZ = "blink_stats.npz"
BLINKING_THREASHOLD = 0.50


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
        "ratio_avg",
        "v_left",
        "h_left",
        "v_right",
        "h_right",
    ]


# Constants for different models
class Model_Constants:
    NUM_FEATURES = 7

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


class Training_Constnats:
    SEQUENCE_LENGTH = 30
    SPLIT_RATIO = 0.6
    CSV_PATH   = r"C:/Users/camil/OneDrive/Programming/Smart_Glass/dev/blink_data.csv"
    BATCH_SIZE = 32
    CURRENT_BEST_F1 = 0.01
