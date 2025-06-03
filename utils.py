import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import fastrcnn_loss

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

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")  # shape: (N,)
        pt = torch.exp(-ce_loss)  # pt = softmax probability of the true class

        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
        

def custom_fastrcnn_loss(class_logits, box_regression, labels, regression_targets, focal_criterion):
    # Flatten inputs
    labels = torch.cat(labels, dim=0)
    class_logits = class_logits.view(-1, class_logits.shape[-1])

    #focal_criterion = FocalLoss(alpha=1.0, gamma=2.0)

    # Apply Focal Loss for classification
    loss_classifier = focal_criterion(class_logits, labels)

    # Smooth L1 loss for box regression (unchanged)
    box_regression = box_regression.view(-1, box_regression.shape[-1])
    regression_targets = torch.cat(regression_targets, dim=0)
    loss_box_reg = F.smooth_l1_loss(box_regression, regression_targets, beta=1.0, reduction='sum') / labels.numel()

    return loss_classifier, loss_box_reg