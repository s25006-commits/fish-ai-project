import torch
from torchvision import transforms, models
from PIL import Image
import os
import torch.nn.functional as F
from fish_info import get_fish_info

# ✅ 图片预处理
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# ❗❗❗ 不要再用 dataset
classes = [
    "amago",
    "ayu",
    "blackbass",
    "bluegill",
    "carp",
    "funa",
    "kawamutsu",
    "nigoi",
    "nijimasu",
    "oikawa",
    "oyanirami"
]

# ❗ 延迟加载
model = None

def load_model():
    global model
    if model is None:
        print("🔥 加载模型中...")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "best_model.pth")

        model = models.resnet18()
        model.fc = torch.nn.Linear(model.fc.in_features, len(classes))

        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()

        print("✅ 模型加载完成")

def predict_image(img_path):
    print("图片路径:", img_path)
    load_model()  # ❗这里才加载模型

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, predicted = torch.max(probs, 1)

    label = classes[predicted.item()]
    info = get_fish_info(label)

    return label, conf.item(), info




