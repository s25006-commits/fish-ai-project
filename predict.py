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
    "namazu",
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
        model.fc = torch.nn.Linear(
            model.fc.in_features,
            len(classes)
        )

        model.load_state_dict(
            torch.load(model_path, map_location="cpu")
        )

        model.eval()

        print("✅ 模型加载完成")


def predict_image(img_path):

    print("图片路径:", img_path)

    load_model()

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0)

    with torch.no_grad():

        output = model(image)

        probs = torch.nn.functional.softmax(
            output,
            dim=1
        )

        print("\n===== 各魚種の確率 =====")

        for i, prob in enumerate(probs[0]):

            print(
                f"{classes[i]:12s}: "
                f"{prob.item() * 100:.2f}%"
            )

        print("========================")

        # ============================================================
        # Top 1 / Top 2
        # ============================================================

        sorted_probs, sorted_indices = torch.sort(
            probs[0],
            descending=True
        )

        # Top1
        confidence = sorted_probs[0].item()

        # Top2
        second_confidence = sorted_probs[1].item()

        # 識別された魚
        predicted_index = sorted_indices[0].item()

        label = classes[predicted_index]

        # Top1 と Top2 の差
        margin = confidence - second_confidence

        print(
            f"Top1: {label} "
            f"{confidence:.2%}"
        )

        print(
            f"Top2: "
            f"{classes[sorted_indices[1].item()]} "
            f"{second_confidence:.2%}"
        )

        print(
            f"Margin: {margin:.2%}"
        )

        # ============================================================
        # ⭐ 不明魚種判定
        # ============================================================

        UNKNOWN_THRESHOLD = 0.60
        UNKNOWN_MARGIN = 0.20

        if (
            confidence < UNKNOWN_THRESHOLD
            and margin < UNKNOWN_MARGIN
        ):

            print("⚠️ 不明魚種として判定")

            return "unknown", confidence, margin, {

                "名前": "不明",

                "外見":
                    "システムに登録されていない魚種の可能性があります。",

                "習性":
                    "現在登録されている魚種以外の可能性があります。",

                "type": "不明"
            }

        # ============================================================
        # 🐟 正常な魚種
        # ============================================================

        info = get_fish_info(label)

        print("識別結果:", label)
        print("精度:", confidence)
        print("Margin:", margin)
        print("魚情報:", info)

        # ⭐ confidence + margin を返す
        return label, confidence, margin, info


# ============================================================
# テスト
# ============================================================

if __name__ == "__main__":

    test_dir = "dataset/test/kawamutsu"

    for filename in os.listdir(test_dir):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        ):

            img_path = os.path.join(
                test_dir,
                filename
            )

            print("\n" + "=" * 60)
            print("测试图片:", filename)
            print("=" * 60)

            predict_image(img_path)




