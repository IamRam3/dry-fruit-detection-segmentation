import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

# Helper to convert predictions to COCO format
def convert_to_coco_format(predictions, image_ids):
    coco_results = []
    for prediction, image_id in zip(predictions, image_ids):
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        for box, score, label in zip(boxes, scores, labels):
            x_min, y_min, x_max, y_max = box
            coco_box = [x_min, y_min, x_max - x_min, y_max - y_min]  # COCO format: [x, y, width, height]
            coco_results.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [float(v) for v in coco_box],
                "score": float(score)
            })
    return coco_results

def get_detection_model(num_classes, segmentation=False):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if segmentation:
        # If segmentation is required, replace the mask predictor
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask, 256, num_classes
        )
    

    return model
