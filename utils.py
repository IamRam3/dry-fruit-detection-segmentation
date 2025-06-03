import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import fastrcnn_loss
import cv2
import os
from natsort import natsorted
from pycocotools.coco import COCO
from tqdm import tqdm
import json
from pycocotools.cocoeval import COCOeval

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

def get_detection_model(num_classes, segmentation=False, custom_focal_loss=False):
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
    
    if custom_focal_loss:
        # Replace the default loss function with a custom focal loss
        model.roi_heads.fastrcnn_loss = custom_fastrcnn_loss
        #model.roi_heads.focal_criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

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


def create_video_from_images(image_folder, output_path, fps=30):
    images = [img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))]
    images = natsorted(images)  # Sort naturally (e.g., img1, img2, ..., img10)

    if not images:
        print("No images found in the directory.")
        return

    # Read the first image to get frame size
    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_image.shape

    # Define the codec and initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for image_name in images:
        img_path = os.path.join(image_folder, image_name)
        frame = cv2.imread(img_path)
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path}")


def train(num_epochs, train_data_loader, model, val_data_loader, convert_to_coco_format, dataset
, device, optimizer, save_path="best_model.pth", patience=5, best_map=0.0):
  for epoch in range(num_epochs):
      model.train()
      total_loss = 0.0

      for images, targets in tqdm(train_data_loader, desc=f"Training Epoch {epoch+1}"):
          images = list(image.to(device) for image in images)
          targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

          loss_dict = model(images, targets)
          losses = sum(loss for loss in loss_dict.values())

          optimizer.zero_grad()
          losses.backward()
          optimizer.step()

          total_loss += losses.item()

      avg_loss = total_loss / len(train_data_loader)
      print(f"[Epoch {epoch+1}] Training Loss: {avg_loss:.4f}")

      # ---- VALIDATION ----
      model.eval()
      predictions = []
      image_ids = []

      with torch.no_grad():
          for images, targets in tqdm(val_data_loader, desc="Validating"):
              images = list(image.to(device) for image in images)
              outputs = model(images)
              predictions.extend(outputs)
              image_ids.extend([int(t["image_id"]) for t in targets])

      coco_predictions = convert_to_coco_format(predictions, image_ids)

      os.makedirs("coco_eval", exist_ok=True)
      prediction_file = "coco_eval/predictions.json"
      with open(prediction_file, "w") as f:
          json.dump(coco_predictions, f)

      coco_gt = COCO(dataset.val_annFile)
      coco_dt = coco_gt.loadRes(prediction_file)
      coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
      coco_eval.evaluate()
      coco_eval.accumulate()
      print(f"[Epoch {epoch+1}] Evaluation Metrics:")
      coco_eval.summarize()

      # --- EARLY STOPPING AND BEST MODEL CHECK ---
      current_map = coco_eval.stats[0]  # AP@[IoU=0.50:0.95]
      if current_map > best_map:
          print(f"New best mAP: {current_map:.4f} (previous best: {best_map:.4f}) — saving model.")
          best_map = current_map
          torch.save(model.state_dict(), save_path)
          epochs_no_improve = 0
      else:
          epochs_no_improve += 1
          print(f"No improvement in mAP for {epochs_no_improve} epoch(s).")

      if epochs_no_improve >= patience:
          print(f"Early stopping triggered after {epoch+1} epochs. Best mAP: {best_map:.4f}")
          break
