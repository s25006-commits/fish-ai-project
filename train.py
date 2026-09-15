import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset

from PIL import Image

import random
from collections import Counter


# ============================================================
# 1. 基本设置
# ============================================================

DATASET_DIR = "dataset"

BATCH_SIZE = 16

EPOCHS = 30

# ⭐ 不再15倍，降低对原始素材的过拟合
AUGMENT_TIMES = 12

# ⭐ 只针对虹鳟增加训练机会
CLASS_AUGMENT_TIMES = {
    "nijimasu": 18
}
TRAIN_RATIO = 0.8

random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


# ============================================================
# 2. 数据增强
# ============================================================

train_transform = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.RandomResizedCrop(
        224,
        scale=(0.80, 1.0)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        15
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.08, 0.08),
        scale=(0.90, 1.10)
    ),

    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        hue=0.05
    ),

    # ⭐ 偶尔降低颜色依赖
    transforms.RandomGrayscale(
        p=0.05
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# 3. 验证集
# ============================================================

val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# 4. 读取原始数据
# ============================================================



print("================================")
print("当前工作目录：", os.getcwd())
print("dataset实际位置：", os.path.abspath(DATASET_DIR))
print("================================")

base_dataset = datasets.ImageFolder(DATASET_DIR)

print("ImageFolder读取图片数量：", len(base_dataset.samples))

missing_files = []

for path, label in base_dataset.samples:
    if not os.path.isfile(path):
        missing_files.append(path)

print("不存在的图片数量：", len(missing_files))

for path in missing_files:
    print("❌ 找不到：", path)
print("\n===== 检查 nijimasu 文件夹 =====")

nijimasu_dir = os.path.abspath(
    os.path.join(DATASET_DIR, "nijimasu")
)

print("实际文件夹：", nijimasu_dir)

files = os.listdir(nijimasu_dir)

print("文件夹实际文件数量：", len(files))

for filename in files:
    print(filename)
print("================================")

print("================================")
print("鱼种类别：")
print(base_dataset.classes)
print("================================")

num_classes = len(base_dataset.classes)

print(
    "原始图片数量：",
    len(base_dataset)
)


# ============================================================
# 5. 检查每种鱼的图片数量
# ============================================================

class_counts = Counter(
    base_dataset.targets
)

print()
print("===== 各鱼种原始图片数量 =====")

for class_id, class_name in enumerate(
    base_dataset.classes
):

    print(
        f"{class_name:12s} : "
        f"{class_counts[class_id]} 张"
    )


# ============================================================
# 6. 每个类别分别进行 80% / 20% 划分
# ============================================================

train_indices = []
val_indices = []


for class_id in range(num_classes):

    class_indices = [

        i

        for i, label in enumerate(
            base_dataset.targets
        )

        if label == class_id
    ]

    random.shuffle(
        class_indices
    )

    train_size = int(
        len(class_indices)
        * TRAIN_RATIO
    )

    train_indices.extend(
        class_indices[:train_size]
    )

    val_indices.extend(
        class_indices[train_size:]
    )


print()
print("训练原图：", len(train_indices))
print("验证原图：", len(val_indices))


# ============================================================
# 7. 自定义 Dataset
# ============================================================

class AugmentedDataset(Dataset):

    def __init__(
        self,
        dataset,
        indices,
        transform,
        default_augment_times=12,
        class_augment_times=None
    ):

        self.dataset = dataset
        self.indices = indices
        self.transform = transform

        self.default_augment_times = default_augment_times
        self.class_augment_times = class_augment_times or {}

        # ⭐ 每张图片按照所属类别决定增强次数
        self.samples = []

        for real_index in self.indices:

            label = self.dataset.targets[real_index]

            class_name = self.dataset.classes[label]

            times = self.class_augment_times.get(
                class_name,
                self.default_augment_times
            )

            for _ in range(times):
                self.samples.append(real_index)


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        real_index = self.samples[index]

        path, label = self.dataset.samples[
            real_index
        ]

        image = Image.open(path).convert("RGB")

        image = self.transform(image)

        return image, label


# ============================================================
# 8. 创建 Dataset
# ============================================================

train_dataset = AugmentedDataset(
    base_dataset,
    train_indices,
    train_transform,
    default_augment_times=AUGMENT_TIMES,
    class_augment_times=CLASS_AUGMENT_TIMES
)


# ⭐ 验证集不做增强
val_dataset = datasets.ImageFolder(
    DATASET_DIR,
    transform=val_transform
)

val_dataset.samples = [
    base_dataset.samples[i]
    for i in val_indices
]

val_dataset.targets = [
    base_dataset.targets[i]
    for i in val_indices
]


# ============================================================
# 9. DataLoader
# ============================================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    num_workers=0
)


val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


# ============================================================
# 10. 计算类别权重
# ============================================================

train_class_counts = Counter(
    base_dataset.targets[i]
    for i in train_indices
)


print()
print("===== 训练集各类别数量 =====")

for class_id in range(num_classes):

    print(
        f"{base_dataset.classes[class_id]:12s} : "
        f"{train_class_counts[class_id]} 张"
    )


# ⭐ 数量越少，权重越高
total_train = len(train_indices)

class_weights = []

for class_id in range(num_classes):

    count = train_class_counts[class_id]

    weight = total_train / (
            num_classes * count
    )

    # ⭐ 重点提高虹鳟的学习权重
    if base_dataset.classes[class_id] == "nijimasu":
        weight *= 1.25

    class_weights.append(weight)


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)


print()
print("===== 类别权重 =====")

for i, weight in enumerate(
    class_weights
):

    print(
        f"{base_dataset.classes[i]:12s} : "
        f"{weight:.3f}"
    )


# ============================================================
# 11. ResNet18
# ============================================================

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# ============================================================
# 12. 先冻结全部网络
# ============================================================

for param in model.parameters():

    param.requires_grad = False


# ============================================================
# 13. ⭐ 解冻 layer3 + layer4
# ============================================================

for param in model.layer3.parameters():

    param.requires_grad = True


for param in model.layer4.parameters():

    param.requires_grad = True


# ============================================================
# 14. 修改最后分类层
# ============================================================

model.fc = nn.Linear(

    model.fc.in_features,

    num_classes
)


# ⭐ fc 必须训练
for param in model.fc.parameters():

    param.requires_grad = True


# ============================================================
# 15. Loss
# ============================================================

criterion = nn.CrossEntropyLoss(

    weight=class_weights,

    label_smoothing=0.05

)


# ============================================================
# 16. Optimizer
# ============================================================

optimizer = optim.Adam(

    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),

    lr=0.0001,

    weight_decay=1e-4

)


# ============================================================
# 17. 学习率调整
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=3
)


# ============================================================
# 18. 训练
# ============================================================

best_val_loss = float("inf")


for epoch in range(EPOCHS):


    # ========================================================
    # Training
    # ========================================================

    model.train()

    running_loss = 0.0

    train_correct = 0

    train_total = 0


    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )


        optimizer.zero_grad()

        loss.backward()

        optimizer.step()


        running_loss += loss.item()


        _, predicted = torch.max(
            outputs,
            1
        )


        train_total += labels.size(0)

        train_correct += (
            predicted == labels
        ).sum().item()


    train_loss = (
        running_loss
        / len(train_loader)
    )


    train_accuracy = (
        train_correct
        / train_total
    )


    # ========================================================
    # Validation
    # ========================================================

    model.eval()

    val_loss = 0.0

    val_correct = 0

    val_total = 0


    # ⭐ 每个鱼种分别统计
    class_correct = [
        0
        for _ in range(num_classes)
    ]

    class_total = [
        0
        for _ in range(num_classes)
    ]


    with torch.no_grad():

        for images, labels in val_loader:

            outputs = model(images)


            loss = criterion(
                outputs,
                labels
            )


            val_loss += loss.item()


            _, predicted = torch.max(
                outputs,
                1
            )


            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


            # ⭐ 每个类别准确率
            for label, pred in zip(
                labels,
                predicted
            ):

                label_id = label.item()

                class_total[label_id] += 1

                if label == pred:

                    class_correct[
                        label_id
                    ] += 1


    val_loss = (
        val_loss
        / len(val_loader)
    )


    val_accuracy = (
        val_correct
        / val_total
    )


    # ========================================================
    # 学习率调整
    # ========================================================

    scheduler.step(
        val_loss
    )


    # ========================================================
    # 输出
    # ========================================================

    print()

    print(
        f"Epoch "
        f"{epoch + 1}/{EPOCHS} | "

        f"Train Loss: "
        f"{train_loss:.4f} | "

        f"Train Acc: "
        f"{train_accuracy:.2%} | "

        f"Val Loss: "
        f"{val_loss:.4f} | "

        f"Val Acc: "
        f"{val_accuracy:.2%}"
    )


    # ========================================================
    # ⭐ 每种鱼的准确率
    # ========================================================

    print(
        "----- 各鱼种 Val Acc -----"
    )


    for class_id in range(
        num_classes
    ):

        if class_total[class_id] > 0:

            acc = (
                class_correct[class_id]
                / class_total[class_id]
            )

            print(
                f"{base_dataset.classes[class_id]:12s}"
                f": {acc:.2%}"
            )


    # ========================================================
    # 保存最佳模型
    # ========================================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss


        torch.save(

            model.state_dict(),

            "best_model.pth"

        )


        print(
            "✅ 最佳模型已保存！"
        )


# ============================================================
# 19. 完成
# ============================================================

print()
print("================================")
print("训练完成！")
print("================================")

print(
    "模型已保存：best_model.pth"
)

print(
    "最佳 Val Loss：",
    best_val_loss
)

