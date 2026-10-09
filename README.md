# ISL Translator

Real-Time Indian Sign Language Detection, Word Formation & Speech Assistant.

This project is a modular Python application for live webcam sign recognition. It uses OpenCV for capture and UI overlays, MediaPipe Hands for hand detection and bounding-box extraction, a pretrained Keras/TensorFlow CNN for recognition, temporal smoothing for stable outputs, word and sentence formation, spelling correction, and offline text-to-speech.

## Model Files

The `models/` directory is intentionally left for you to populate manually. Add:

- `models/isl_model.h5`
- `models/labels.json`

Optional:

- `models/model_info.json`

For the current pretrained CNN, the expected input is grayscale `128x128x1` and the output is `35` classes with softmax.

The shipped label order is:

- `1` to `9`
- `A` to `Z`

If your model artifact contains a `Lambda` layer, the loader will attempt unsafe deserialization because this is a trusted local offline model.

`labels.json` may be either:

```json
["1", "2", "3"]
```

or:

```json
{"0": "1", "1": "2", "2": "3"}
```

`model_info.json` is only needed if you want to record metadata for future variants. The current application already knows how to preprocess the inspected CNN model:

- crop the hand ROI
- convert to grayscale
- resize to `128x128`
- normalize to `[0, 1]`
- expand to `(1, 128, 128, 1)`

## Run

```powershell
python main.py
```

Controls:

- `Q` quit
- `S` speak current sentence or word
- `Enter` commit the current word with spell correction
- `Space` commit word or add a space
- `Backspace` delete the last character or previous word
- `C` clear all text
