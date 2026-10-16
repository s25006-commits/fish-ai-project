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

# ⭐ 现在不再直接读取 dataset/
# ⭐ 而是明确区分 Train / Val / Test

TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
TEST_DIR = "dataset/test"

BATCH_SIZE = 16

EPOCHS = 30

# ⭐ 每张训练图片进行多少次随机增强
AUGMENT_TIMES = 12

# ⭐ 暂时不针对任何鱼额外增加训练次数
CLASS_AUGMENT_TIMES = {}


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

    # ⭐ 偶尔降低模型对颜色的依赖
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
# 3. Val / Test 数据处理
# ============================================================

# ⭐ Val 和 Test 都不能进行随机增强
# ⭐ 保证每次评价结果稳定

val_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


test_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.224]
    )
])


# ============================================================
# 4. 检查数据集路径
# ============================================================

print("================================")
print("当前工作目录：")
print(os.getcwd())
print("================================")

print("Train 数据集：")
print(os.path.abspath(TRAIN_DIR))

print()

print("Val 数据集：")
print(os.path.abspath(VAL_DIR))

print()

print("Test 数据集：")
print(os.path.abspath(TEST_DIR))

print("================================")


if not os.path.exists(TRAIN_DIR):

    raise FileNotFoundError(
        f"❌ 找不到训练集：{os.path.abspath(TRAIN_DIR)}"
    )


if not os.path.exists(VAL_DIR):

    raise FileNotFoundError(
        f"❌ 找不到验证集：{os.path.abspath(VAL_DIR)}"
    )


if not os.path.exists(TEST_DIR):

    raise FileNotFoundError(
        f"❌ 找不到测试集：{os.path.abspath(TEST_DIR)}"
    )


# ============================================================
# 5. 读取 Train / Val / Test
# ============================================================

train_base_dataset = datasets.ImageFolder(
    TRAIN_DIR
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR,
    transform=test_transform
)
from collections import Counter

print("\n===== Train 各类别数量 =====")
print(Counter(train_base_dataset.targets))

print("\n===== Val 各类别数量 =====")
print(Counter(val_dataset.targets))

print("\n===== Test 各类别数量 =====")
print(Counter(test_dataset.targets))

print("\n===== 类别顺序 =====")
print(train_base_dataset.class_to_idx)
print(val_dataset.class_to_idx)
print(test_dataset.class_to_idx)

from collections import Counter

print("\n================================")
print("数据集检查")
print("================================")

print("\n----- Train -----")
print(train_base_dataset.class_to_idx)
print(Counter(train_base_dataset.targets))

print("\n----- Val -----")
print(val_dataset.class_to_idx)
print(Counter(val_dataset.targets))

print("\n----- Test -----")
print(test_dataset.class_to_idx)
print(Counter(test_dataset.targets))

print("================================\n")

# ============================================================
# 6. 检查三个数据集的类别是否完全一致
# ============================================================

print()
print("================================")
print("检查数据集类别")
print("================================")

print("Train 类别：")
print(train_base_dataset.classes)

print()

print("Val 类别：")
print(val_dataset.classes)

print()

print("Test 类别：")
print(test_dataset.classes)


if train_base_dataset.classes != val_dataset.classes:

    raise ValueError(
        "❌ Train 和 Val 的鱼种类别不一致！"
    )


if train_base_dataset.classes != test_dataset.classes:

    raise ValueError(
        "❌ Train 和 Test 的鱼种类别不一致！"
    )


print()
print("✅ Train / Val / Test 类别一致")
print("================================")


# ============================================================
# 7. 类别数量
# ============================================================

classes = train_base_dataset.classes

num_classes = len(classes)

print()
print("鱼种类别数量：", num_classes)

print()
print("===== 鱼种类别 =====")

for i, class_name in enumerate(classes):

    print(
        f"{i}: {class_name}"
    )


# ============================================================
# 8. 检查图片数量
# ============================================================

print()
print("================================")
print("数据集图片数量")
print("================================")

print(
    "Train:",
    len(train_base_dataset)
)

print(
    "Val  :",
    len(val_dataset)
)

print(
    "Test :",
    len(test_dataset)
)

print("================================")


# ============================================================
# 9. 检查每种鱼的图片数量
# ============================================================

train_class_counts = Counter(
    train_base_dataset.targets
)

val_class_counts = Counter(
    val_dataset.targets
)

test_class_counts = Counter(
    test_dataset.targets
)


print()
print("===== Train 各鱼种数量 =====")

for class_id, class_name in enumerate(classes):

    print(
        f"{class_name:12s} : "
        f"{train_class_counts[class_id]} 张"
    )


print()
print("===== Val 各鱼种数量 =====")

for class_id, class_name in enumerate(classes):

    print(
        f"{class_name:12s} : "
        f"{val_class_counts[class_id]} 张"
    )


print()
print("===== Test 各鱼种数量 =====")

for class_id, class_name in enumerate(classes):

    print(
        f"{class_name:12s} : "
        f"{test_class_counts[class_id]} 张"
    )


# ============================================================
# 10. 自定义 Dataset
# ============================================================

class AugmentedDataset(Dataset):

    def __init__(
        self,
        dataset,
        transform,
        default_augment_times=12,
        class_augment_times=None
    ):

        self.dataset = dataset

        self.transform = transform

        self.default_augment_times = (
            default_augment_times
        )

        self.class_augment_times = (
            class_augment_times or {}
        )

        # ⭐ 保存实际训练样本索引
        self.samples = []

        for real_index in range(
            len(self.dataset.samples)
        ):

            label = self.dataset.targets[
                real_index
            ]

            class_name = self.dataset.classes[
                label
            ]

            times = self.class_augment_times.get(
                class_name,
                self.default_augment_times
            )

            for _ in range(times):

                self.samples.append(
                    real_index
                )


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        real_index = self.samples[
            index
        ]

        path, label = self.dataset.samples[
            real_index
        ]

        image = Image.open(
            path
        ).convert("RGB")

        image = self.transform(
            image
        )

        return image, label


# ============================================================
# 11. 创建训练 Dataset
# ============================================================

train_dataset = AugmentedDataset(

    train_base_dataset,

    train_transform,

    default_augment_times=AUGMENT_TIMES,

    class_augment_times=CLASS_AUGMENT_TIMES

)


print()
print("================================")
print("训练数据增强")
print("================================")

print(
    "原始 Train 图片：",
    len(train_base_dataset)
)

print(
    "增强后 Train 图片：",
    len(train_dataset)
)

print(
    "每张原图默认增强次数：",
    AUGMENT_TIMES
)

print("================================")


# ============================================================
# 12. DataLoader
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


test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    num_workers=0
)


# ============================================================
# 13. 计算类别权重
# ============================================================

# ⭐ 注意：
# ⭐ 权重只根据原始 Train 图片数量计算
# ⭐ 不能根据增强后的数量计算

total_train = len(
    train_base_dataset
)

class_weights = []


for class_id in range(
    num_classes
):

    count = train_class_counts[
        class_id
    ]

    weight = total_train / (
        num_classes * count
    )

    class_weights.append(
        weight
    )


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
        f"{classes[i]:12s} : "
        f"{weight:.3f}"
    )


# ============================================================
# 14. ResNet18
# ============================================================

print()
print("================================")
print("加载 ResNet18")
print("================================")


model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)


# ============================================================
# 15. 冻结全部网络
# ============================================================

for param in model.parameters():

    param.requires_grad = False


# ============================================================
# 16. 解冻 layer3 + layer4
# ============================================================

for param in model.layer3.parameters():

    param.requires_grad = True


for param in model.layer4.parameters():

    param.requires_grad = True


# ============================================================
# 17. 修改最后分类层
# ============================================================

model.fc = nn.Linear(

    model.fc.in_features,

    num_classes
)


# ⭐ fc 必须训练

for param in model.fc.parameters():

    param.requires_grad = True


print()
print("Trainable Layers:")
print("layer3")
print("layer4")
print("fc")


# ============================================================
# 18. Loss
# ============================================================

criterion = nn.CrossEntropyLoss(

    weight=class_weights,

    label_smoothing=0.05

)


# ============================================================
# 19. Optimizer
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
# 20. 学习率调整
# ============================================================

scheduler = optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=3

)


# ============================================================
# 21. 开始训练
# ============================================================

print()
print("================================")
print("开始训练")
print("================================")


best_val_loss = float("inf")


for epoch in range(
    EPOCHS
):


    # ========================================================
    # Training
    # ========================================================

    model.train()

    running_loss = 0.0

    train_correct = 0

    train_total = 0


    for images, labels in train_loader:

        outputs = model(
            images
        )

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

            outputs = model(
                images
            )


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


            # ⭐ 每个鱼种统计

            for label, pred in zip(
                labels,
                predicted
            ):

                label_id = label.item()

                class_total[
                    label_id
                ] += 1


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
    # 输出 Epoch
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
    # 各鱼种 Val Accuracy
    # ========================================================

    print(
        "----- 各鱼种 Val Acc -----"
    )


    for class_id in range(
        num_classes
    ):

        if class_total[
            class_id
        ] > 0:

            acc = (
                class_correct[
                    class_id
                ]
                /
                class_total[
                    class_id
                ]
            )

            print(
                f"{classes[class_id]:12s}"
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
# 22. 训练完成
# ============================================================

print()
print("================================")
print("训练完成！")
print("================================")

print(
    "最佳 Val Loss：",
    best_val_loss
)

print(
    "模型已保存：best_model.pth"
)


# ============================================================
# 23. 加载最佳模型
# ============================================================

print()
print("================================")
print("加载最佳模型进行 Test")
print("================================")


model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location="cpu"
    )
)

model.eval()


# ============================================================
# 24. Test
# ============================================================

test_correct = 0

test_total = 0


test_class_correct = [
    0
    for _ in range(num_classes)
]


test_class_total = [
    0
    for _ in range(num_classes)
]


with torch.no_grad():

    for images, labels in test_loader:

        outputs = model(
            images
        )


        _, predicted = torch.max(
            outputs,
            1
        )


        test_total += labels.size(0)


        test_correct += (
            predicted == labels
        ).sum().item()


        # ⭐ 各鱼种 Test Accuracy

        for label, pred in zip(
            labels,
            predicted
        ):

            label_id = label.item()

            test_class_total[
                label_id
            ] += 1


            if label == pred:

                test_class_correct[
                    label_id
                ] += 1


# ============================================================
# 25. Test Accuracy
# ============================================================

test_accuracy = (
    test_correct
    / test_total
)


print()
print("================================")
print("最终 Test 结果")
print("================================")

print(
    f"Test Accuracy: "
    f"{test_accuracy:.2%}"
)


print()
print(
    "----- 各鱼种 Test Accuracy -----"
)


for class_id in range(
    num_classes
):

    if test_class_total[
        class_id
    ] > 0:

        acc = (
            test_class_correct[
                class_id
            ]
            /
            test_class_total[
                class_id
            ]
        )

        print(
            f"{classes[class_id]:12s}"
            f": {acc:.2%}"
        )


print()
print("================================")
print("Test 完成")
print("================================")

