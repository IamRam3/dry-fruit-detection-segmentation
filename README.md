## See 'Main.ipynb' for Usage or the Code in Action.

## 📊 Model Performance

| Model Variant      | Task        | mAP@[0.5:0.95] | Inference time | Size | Comments               |
|--------------------|-------------|----------------|----------------|------|------------------------|
| Faster R-CNN       | Detection   | 0.7526         |                |      |Accurate but heavier    |
| MobileNet FRCNN    | Detection   | 0.7418         |                |      |Lightweight, faster     |
| Mask R-CNN         | Segmentation| 0.7329         |                |      |Good mask alignment     |

## To-Do:
- [ ] ONNX to TFLite conversion
