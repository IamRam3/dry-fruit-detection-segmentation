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
import torchvision.transforms as T
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
from natsort import natsorted

def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

# Helper to convert predictions to COCO format
def convert_to_coco_format(predictions, image_ids, segmentation=False):
    coco_results = []
    for prediction, image_id in zip(predictions, image_ids):
        boxes = prediction["boxes"].cpu().numpy()
        scores = prediction["scores"].cpu().numpy()
        labels = prediction["labels"].cpu().numpy()
        if segmentation:
            masks = prediction["masks"].cpu().numpy()  # shape: [N, 1, H, W]
            for box, score, label, mask in zip(boxes, scores, labels, masks):
                x_min, y_min, x_max, y_max = box
                coco_box = [x_min, y_min, x_max - x_min, y_max - y_min]

            # Threshold mask
                mask = mask[0] > 0.5  # Convert from [1, H, W] to binary mask

            # Encode mask in RLE
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")  # for JSON serializable

                coco_results.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [float(v) for v in coco_box],
                    "score": float(score),
                    "segmentation": rle
                })
        else:
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

def get_detection_model(num_classes, mobilenet=False, custom_focal_loss=False):

    if mobilenet:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    else:
    # load an instance segmentation model pre-trained pre-trained on COCO
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if custom_focal_loss:
        # Replace the default loss function with a custom focal loss
        model.roi_heads.fastrcnn_loss = custom_fastrcnn_loss
        #model.roi_heads.focal_criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
    
    return model

def get_segmentation_model(num_classes, custom_focal_loss=False):

    model = maskrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if custom_focal_loss:
        # Replace the default loss function with a custom focal loss
        model.roi_heads.fastrcnn_loss = custom_fastrcnn_loss
        #model.roi_heads.focal_criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction="mean")

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
        

def custom_fastrcnn_loss(class_logits, box_regression, labels, regression_targets, focal_criterion = FocalLoss(alpha=1.0, gamma=2.0)):
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
, device, optimizer, save_path="best_model.pth", patience=5, best_map=0.0, segmentation=False):
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

      coco_predictions = convert_to_coco_format(predictions, image_ids, segmentation)

      if len(coco_predictions) == 0:
            print("[Warning] No valid predictions generated. Skipping COCO evaluation.")
            continue  # Or safely skip this epoch’s evaluation

      os.makedirs("coco_eval", exist_ok=True)
      prediction_file = "coco_eval/predictions.json"
      with open(prediction_file, "w") as f:
          json.dump(coco_predictions, f)

      coco_gt = COCO(dataset.val_annFile)
      coco_dt = coco_gt.loadRes(prediction_file)
      if segmentation:
          iouType = 'segm'
      else:
          iouType = 'bbox'
      coco_eval = COCOeval(coco_gt, coco_dt, iouType=iouType)
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


def save_predicted_image(img_path, output_path, model, device, id_to_name, threshold=0.5):
    image = Image.open(img_path).convert("RGB")
    transform = T.ToTensor()
    image_tensor = transform(image).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model([image_tensor])[0]

    boxes = outputs["boxes"].cpu()
    scores = outputs["scores"].cpu()
    labels = outputs["labels"].cpu()

    # Convert PIL to OpenCV format (RGB to BGR)
    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for i in range(len(scores)):
        if scores[i] >= threshold:
            box = boxes[i].numpy().astype(int)
            label = id_to_name.get(labels[i].item(), str(labels[i].item()))
            score = scores[i].item()

            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {score:.2f}", (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imwrite(output_path, frame)


def save_predicted_video(video_path, output_path, model, device, id_to_name, threshold=0.5):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    transform = T.ToTensor()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_tensor = transform(frame).to(device)

        model.to(device)
        model.eval()
        with torch.no_grad():
            outputs = model([image_tensor])[0]

        boxes = outputs["boxes"].cpu()
        scores = outputs["scores"].cpu()
        labels = outputs["labels"].cpu()

        for i in range(len(scores)):
            if scores[i] >= threshold:
                box = boxes[i].numpy().astype(int)
                label = id_to_name.get(labels[i].item(), str(labels[i].item()))
                score = scores[i].item()

                cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {score:.2f}", (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()


def closest_basic_color_name(requested_color, basic_colors):
    min_dist = float("inf")
    closest_name = None
    for name, rgb in basic_colors.items():
        dist = sum((rc - cc) ** 2 for rc, cc in zip(requested_color, rgb))
        if dist < min_dist:
            min_dist = dist
            closest_name = name
    return closest_name

def detect_shape(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"

    # Use the largest contour
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 100:  # too small to be reliable
        return "tiny"

    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * perimeter, True)

    # Shape classification
    if len(approx) >= 6:
        # Use circularity = 4π * Area / Perimeter²
        circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-5)
        if circularity > 0.7:
            return "round"
        elif circularity > 0.4:
            return "oval"
        else:
            return "irregular"
    else:
        return "irregular"
    

def save_segmented_image(img_path, output_path, model, device, id_to_name, threshold=0.5, mask_threshold=0.5):
    image = Image.open(img_path).convert("RGB")
    transform = T.ToTensor()

    # Fixed overlay color (BGR)
    FIXED_COLOR = np.array([0, 0, 255], dtype=np.uint8)  # white color
    image_tensor = transform(image).to(device)

    model.to(device)
    model.eval()
    with torch.no_grad():
        outputs = model([image_tensor])[0]

    boxes = outputs["boxes"].cpu()
    scores = outputs["scores"].cpu()
    labels = outputs["labels"].cpu()
    masks = outputs["masks"].cpu()  # [N, 1, H, W]

    frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    for i in range(len(scores)):
        if scores[i] >= threshold:
            box = boxes[i].numpy().astype(int)
            label = id_to_name.get(labels[i].item(), str(labels[i].item()))
            score = scores[i].item()

            # Get and threshold mask
            mask = masks[i, 0].numpy()
            mask = (mask > mask_threshold).astype(np.uint8)

            shape = detect_shape(mask)

            # Extract masked region from original image to compute average color
            masked_region = frame * mask[:, :, None]
            if np.any(mask):
                avg_color = masked_region[mask.astype(bool)].mean(axis=0).astype(np.uint8)
            else:
                avg_color = np.array([128, 128, 128], dtype=np.uint8)  # fallback

            # Get color name from actual region color (convert to RGB)
            avg_color_rgb = avg_color[::-1]
            color_name = closest_basic_color_name(avg_color)

            # Prepare fixed colored mask for overlay
            colored_mask = np.zeros_like(frame)
            for c in range(3):
                colored_mask[:, :, c] = mask * FIXED_COLOR[c]

            # Overlay fixed-color mask on frame
            frame = cv2.addWeighted(frame, 0.7, colored_mask, 0.5, 0)

            # Draw box and label (color of text = original avg color)
            label_text = f"{label} {score:.2f} - {color_name.capitalize()}, {shape.capitalize()}"
            cv2.rectangle(frame, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
            cv2.putText(frame, label_text, (box[0], box[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (int(avg_color[0]), int(avg_color[1]), int(avg_color[2])), 2)

    cv2.imwrite(output_path, frame)
    print(f"[✓] Saved segmented image to: {output_path}")


def get_all_image_paths(folder_path):
    valid_exts = (".jpg", ".jpeg", ".png")
    return [os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.lower().endswith(valid_exts)]


def save_segmented_video(video_path, frame_output_dir, threshold=0.5, mask_threshold=0.5):
    if not os.path.exists(frame_output_dir):
        os.makedirs(frame_output_dir)

    cap = cv2.VideoCapture(video_path)
    transform = T.ToTensor()
    frame_idx = 0

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        image_tensor = transform(pil_image).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model([image_tensor])[0]

        frame_display = frame_bgr.copy()  # Start with original BGR for output

        boxes = outputs["boxes"].cpu()
        scores = outputs["scores"].cpu()
        labels = outputs["labels"].cpu()
        masks = outputs["masks"].cpu()  # [N, 1, H, W]

        for i in range(len(scores)):
            if scores[i] >= threshold:
                box = boxes[i].numpy().astype(int)
                label = id_to_name.get(labels[i].item(), str(labels[i].item()))
                score = scores[i].item()

                # Threshold and apply mask
                mask = masks[i, 0].numpy()
                mask = (mask > mask_threshold).astype(np.uint8)

                # Extract masked region and compute average color
                masked_region = frame_display * mask[:, :, None]
                if np.any(mask):
                    color = masked_region[mask.astype(bool)].mean(axis=0).astype(np.uint8)
                else:
                    color = np.array([128, 128, 128], dtype=np.uint8)

                # Get color name and shape
                color_name = closest_basic_color_name(color)
                shape = detect_shape(mask)

                # Create overlay mask
                colored_mask = np.zeros_like(frame_display)
                for c in range(3):
                    colored_mask[:, :, c] = mask * color[c]

                frame_display = cv2.addWeighted(frame_display, 1.0, colored_mask, 0.5, 0)

                # Draw box and label
                label_text = f"{label} {score:.2f} - {color_name.capitalize()}, {shape.capitalize()}"
                cv2.rectangle(frame_display, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 2)
                cv2.putText(frame_display, label_text, (box[0], box[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(color[0]), int(color[1]), int(color[2])), 2)

        # Save processed frame
        frame_name = f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(os.path.join(frame_output_dir, frame_name), frame_display)
        frame_idx += 1

    cap.release()
    print(f"[✓] Saved {frame_idx} segmented frames to: {frame_output_dir}")


class MaskRCNNWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        boxes = outputs[0]["boxes"]
        labels = outputs[0]["labels"]
        scores = outputs[0]["scores"]
        masks = outputs[0]["masks"]
        return boxes, labels, scores, masks
    

class FasterRCNNWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        boxes = outputs[0]["boxes"]
        labels = outputs[0]["labels"]
        scores = outputs[0]["scores"]
        return boxes, labels, scores