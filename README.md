# Smart‑Glasses Blink & Gesture Recognition

A lightweight computer‑vision pipeline that runs **on‑device** (webcam or inward‑facing monocular camera) to detect eye blinks and other micro‑gestures, then relays them to a host computer or phone as configurable macros (e.g. *double‑blink ⇒ pause YouTube*).

The repository currently focuses on reliable **blink detection** as a first use‑case. All core pieces are in place so you can:

- **Collect labelled data** with a single key‑press (`data_factory.py`).
- **Train / fine‑tune** a small PyTorch LSTM (`train_blink.py`).
- **Run real‑time inference** and see results overlaid on live video (`video_blink_test.py`).

Once the blink recogniser is solid, extra gestures (gaze shifts, wink patterns, nods) can be dropped in with the **same pipeline**.

* * *

## Contents

    smart‑glasses/├── data_factory.py # capture video + EAR features + pixel patch to CSV├── train_blink.py # prepare sequences, train LSTM, save .torch model├── model.py # network definition & helpers (imported by train/test)├── video_blink_test.py # run model live on webcam and print trig events├── requirements.txt # pip dependencies└── README.md # you are here

* * *

## Quick‑start

1. **Clone & create env**

        git clone https://github.com/<your‑user>/smart‑glasses.gitcd smart‑glassespython -m venv .venv && source .venv/bin/activate # Windows: .venv\Scripts\activatepip install -r requirements.txt
2. **Collect a few blinks**

        python data_factory.py

    - Press **SPACE** once per blink – a 250 ms window of 1’s is written to the CSV so you don’t have to time it perfectly.
    - Press **Q** to stop. The script prints the CSV filename (e.g. `blink_data_20250604_184055.csv`).
3. **Train / fine‑tune**

        python Model/train_blink.py

    The script reads `dev/blinkdata.csv` and writes `blink_best.pth` plus
    normalisation statistics to `blink_stats.npz` when validation improves.
4. **Live test**

        python video_blink_test.py --model models/blink_lstm_epoch20_acc0.92.torch

    An OpenCV window shows EAR curves, model logits, and debounced blink events.

* * *

## Dependencies

- Python 3.9+
- OpenCV‑Python (4.x)
- **cvzone** ‑ convenience wrapper around MediaPipe
- MediaPipe‑solutions (if not already pulled in by cvzone)
- PyTorch &gt;= 2.0
- numpy, pandas, matplotlib, keyboard

Install everything via `pip install -r requirements.txt`.

> 
> **Note:** `keyboard` needs root/administrator privileges on Linux/macOS to read key events. On Windows it works out of the box.

* * *

## Data format

Each CSV row ≈ one video frame.

| column | description |
| --- | --- |
| `timestamp` | seconds since script start |
| `ratio_left`, `ratio_right`, `ratio_avg` | eye‑aspect‑ratio metrics |
| `v_left`, `h_left`, … | raw vertical/horizontal distances (pixels) |
| `blink_count` | running integer counter (incremented on *rising edge*) |
| `manual_blink` | **label** → 1 during the blink window, else 0 |
| `px_i` | flattened 24×12 left‑eye greyscale patch (288 ints) |

Sequence loader in `train_blink.py` slices these into overlapping windows (`seq_len` frames) and balances positive/negative samples automatically.

* * *

## Training tips

- **Sequence length:** 15–25 frames works well at 30 fps.
- **Class balance:** Script computes `pos_weight` so you rarely need manual re‑sampling.
- **Early stopping:** Use `--patience` to abort if val‑loss stops improving.
- **Augmentation:** Try random brightness/contrast jitter on the eye patch tensor for robustness to lighting.

* * *

## Macro/Automation layer (roadmap)

- MQTT / WebSocket client that publishes recognised gestures.
- Companion app (Electron / mobile) that maps incoming events to OS‑level hotkeys.
- Profile‑based mapping (e.g. *Media* vs *IDE* presets).

* * *

## Contributing

Pull requests are welcome! Please open an issue first to discuss major changes.

- **Bug fixes** – clear reproduction steps help a ton.
- **New gestures** – provide at least one example CSV and a short description.
- **Docs** – typos, clarifications, screenshots – all good.

* * *

## License

This project is released under the **MIT License** – see `LICENSE` for details.

* * *

## Acknowledgements

- [cvzone](https://github.com/cvzone/cvzone) for the FaceMesh & plotting helpers.
- MediaPipe FaceMesh for landmark detection.
- The OpenCV and PyTorch communities for making vision research so accessible.