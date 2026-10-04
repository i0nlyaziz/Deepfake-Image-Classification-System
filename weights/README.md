# Model Weights

The trained weights aren't committed to this repo directly — they're excluded in `.gitignore` and attached to the repo's Releases instead.

## Download

- **best_real_fake_model.pth** — [Download from Releases](https://github.com/i0nlyaziz/Deepfake-Image-Classification-System/releases/download/v1.0/best_real_fake_model.pth)
- **face_detection_yunet_2026may.onnx** — [Download from Releases](https://github.com/i0nlyaziz/Deepfake-Image-Classification-System/releases/download/v1.0/face_detection_yunet_2026may.onnx)

## Using the weights

1. Download both files from the links above and place them in this `weights/` folder.
2. `main.py` loads them from here: `best_real_fake_model.pth` is the ResNet50 classifier, and `face_detection_yunet_2026may.onnx` is the YuNet face detector.
3. The `.pth` file holds the weights for a ResNet50 whose last layer has 2 outputs: class 0 is Fake and class 1 is Real.
4. Update the model paths in `main.py` if you place the files somewhere other than `weights/`.