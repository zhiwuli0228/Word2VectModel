import torch
from torch.optim import AdamW
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# 1. 数据准备
from datasets import Dataset

data = [
    {"image_path": r"data/dog.jpeg", "text": "A dog running on the grass"},
    {"image_path": r"data/cat.jpg", "text": "A cat sitting on the sofa"},
    {"image_path": r"data/car.jpg", "text": "A car speeding on the road"}
]

dataset = Dataset.from_list(data)

# 2. 模型加载
model_name = r"D:\workspace\clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name)


# 3. 数据处理
def preprocess(batch):
    images = [Image.open(d["image_path"]).convert("RGB") for d in batch]
    texts = [d["text"] for d in batch]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
    return inputs


# 4. 定义训练目标
# 5. 训练
optimizer = AdamW(model.parameters(), lr=5e-6)

for epoch in range(2):  # 小规模测试
    inputs = preprocess(data)
    outputs = model(**inputs)

    # CLIP 内置提供对比损失 logits
    logits_per_image = outputs.logits_per_image
    logits_per_text = outputs.logits_per_text

    labels = torch.arange(len(logits_per_image), device=logits_per_image.device)
    loss_img = torch.nn.functional.cross_entropy(logits_per_image, labels)
    loss_txt = torch.nn.functional.cross_entropy(logits_per_text, labels)
    loss = (loss_img + loss_txt) / 2

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Epoch {epoch}, Loss: {loss.item()}")
# 6. 验证&推理
texts = ["a dog", "a cat", "a car"]
images = [Image.open(r"data/dog.jpeg").convert("RGB")]

inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
outputs = model(**inputs)

logits_per_image = outputs.logits_per_image  # 图 -> 文
probs = logits_per_image.softmax(dim=1)
print("匹配概率：", probs)
