import torch
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes=2)

model.load_state_dict(torch.load('artifacts/models/fasterrcnn_epoch10.pth', map_location=device))
model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict_and_draw(image: Image.Image):
    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    prediction = predictions[0]
    boxes = prediction['boxes'].cpu().numpy()
    labels = prediction['labels'].cpu().numpy()
    scores = prediction['scores'].cpu().numpy()

    img_rgb = image.convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    for box, score in zip(boxes, scores):
        if score > 0.7:
            x_min, y_min, x_max, y_max = box
            draw.rectangle([x_min, y_min, x_max, y_max], outline='red', width=2)

    return img_rgb
