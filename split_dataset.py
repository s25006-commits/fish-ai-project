import os
import random
import shutil


SOURCE_DIR = "dataset"


TRAIN_DIR = os.path.join(SOURCE_DIR, "train")
VAL_DIR = os.path.join(SOURCE_DIR, "val")
TEST_DIR = os.path.join(SOURCE_DIR, "test")


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


random.seed(42)


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

# ==========================================
# 图片格式
# ==========================================
IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

# ==========================================
# 检查比例
# ==========================================
if abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) > 1e-6:
    raise ValueError(
        "Train / Val / Test 比例必须加起来等于 1.0"
    )


# ==========================================
# 复制函数
# ==========================================
def safe_copy(src, dst):

    # 检查源文件是否存在
    if not os.path.isfile(src):
        print(f"⚠️ 源文件不存在，跳过：")
        print(f"   {src}")
        return False

    try:

        # 确保目标目录存在
        os.makedirs(
            os.path.dirname(dst),
            exist_ok=True
        )

        shutil.copy2(
            src,
            dst
        )

        return True

    except Exception as e:

        print(f"❌ 复制失败：")
        print(f"   源文件：{src}")
        print(f"   目标文件：{dst}")
        print(f"   错误：{e}")

        return False


# ==========================================
# 开始
# ==========================================

print("================================")
print("开始重新划分数据集")
print("================================")


# ==========================================
# 1. 检查原始数据
# ==========================================

print("\n===== 原始数据检查 =====")

original_counts = {}

for class_name in classes:

    source_class_dir = os.path.join(
        SOURCE_DIR,
        class_name
    )

    if not os.path.isdir(source_class_dir):

        print(
            f"❌ 找不到目录：{source_class_dir}"
        )

        original_counts[class_name] = 0
        continue

    files = [
        f
        for f in os.listdir(source_class_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
        and os.path.isfile(
            os.path.join(source_class_dir, f)
        )
    ]

    original_counts[class_name] = len(files)

    print(
        f"{class_name:12s}: {len(files)} 张"
    )

print("================================")


# ==========================================
# 2. 删除旧 train / val / test
# ==========================================

for folder in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]:

    if os.path.exists(folder):

        print(
            f"\n🗑️ 删除旧目录：{folder}"
        )

        shutil.rmtree(folder)


# ==========================================
# 3. 创建新的 train / val / test
# ==========================================

for folder in [
    TRAIN_DIR,
    VAL_DIR,
    TEST_DIR
]:

    os.makedirs(
        folder,
        exist_ok=True
    )

    for class_name in classes:

        os.makedirs(
            os.path.join(
                folder,
                class_name
            ),
            exist_ok=True
        )


# ==========================================
# 4. 开始划分
# ==========================================

print("\n================================")
print("开始划分")
print("================================")


for class_name in classes:

    source_class_dir = os.path.join(
        SOURCE_DIR,
        class_name
    )

    # --------------------------------------
    # 获取原始图片
    # --------------------------------------

    if not os.path.isdir(source_class_dir):

        print(
            f"\n❌ 跳过 {class_name}："
            f"目录不存在"
        )

        continue

    files = [
        f
        for f in os.listdir(source_class_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
        and os.path.isfile(
            os.path.join(source_class_dir, f)
        )
    ]

    # --------------------------------------
    # 随机打乱
    # --------------------------------------

    random.shuffle(files)

    total = len(files)

    # --------------------------------------
    # 计算数量
    # --------------------------------------

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    test_count = (
        total
        - train_count
        - val_count
    )

    # --------------------------------------
    # 划分
    # --------------------------------------

    train_files = files[
        :train_count
    ]

    val_files = files[
        train_count:
        train_count + val_count
    ]

    test_files = files[
        train_count + val_count:
    ]

    # ======================================
    # 输出
    # ======================================

    print(
        f"\n{class_name}"
    )

    print(
        f"总数   : {total}"
    )

    print(
        f"Train  : {len(train_files)}"
    )

    print(
        f"Val    : {len(val_files)}"
    )

    print(
        f"Test   : {len(test_files)}"
    )

    # ======================================
    # Train
    # ======================================

    for filename in train_files:

        src = os.path.join(
            source_class_dir,
            filename
        )

        dst = os.path.join(
            TRAIN_DIR,
            class_name,
            filename
        )

        safe_copy(
            src,
            dst
        )

    # ======================================
    # Val
    # ======================================

    for filename in val_files:

        src = os.path.join(
            source_class_dir,
            filename
        )

        dst = os.path.join(
            VAL_DIR,
            class_name,
            filename
        )

        safe_copy(
            src,
            dst
        )

    # ======================================
    # Test
    # ======================================

    for filename in test_files:

        src = os.path.join(
            source_class_dir,
            filename
        )

        dst = os.path.join(
            TEST_DIR,
            class_name,
            filename
        )

        safe_copy(
            src,
            dst
        )


# ==========================================
# 5. 最终检查
# ==========================================

print("\n================================")
print("重新划分完成")
print("================================")


grand_total = 0


for folder_name, folder_path in [
    ("Train", TRAIN_DIR),
    ("Val", VAL_DIR),
    ("Test", TEST_DIR)
]:

    print(
        f"\n===== {folder_name} ====="
    )

    total_images = 0

    for class_name in classes:

        class_dir = os.path.join(
            folder_path,
            class_name
        )

        if not os.path.isdir(class_dir):

            count = 0

        else:

            count = len([
                f
                for f in os.listdir(class_dir)
                if f.lower().endswith(
                    IMAGE_EXTENSIONS
                )
                and os.path.isfile(
                    os.path.join(
                        class_dir,
                        f
                    )
                )
            ])

        total_images += count

        print(
            f"{class_name:12s}: {count} 张"
        )

    print(
        f"{folder_name} 总计: {total_images} 张"
    )

    grand_total += total_images


# ==========================================
# 6. 最终总数
# ==========================================

original_total = sum(
    original_counts.values()
)

print("\n================================")
print("最终数据检查")
print("================================")

print(
    f"原始图片总数 : {original_total}"
)

print(
    f"Train + Val + Test : {grand_total}"
)

if grand_total == original_total:

    print(
        "✅ 所有图片都已成功划分"
    )

else:

    print(
        "⚠️ 注意：划分后的图片总数与原始数量不一致"
    )

    print(
        f"少了：{original_total - grand_total} 张"
    )

print("================================")