import torch
import os
from models.backbone.vit import vit_base_patch16  # 你原来的ViT实现
from models.branch_slot.slot_attention import SlotAttention
from models.branch_proto.prototype_bank import MultiPrototypeBank
from models.branch_cs_hps.heads import ContentStyleHeads
from models.branch_cs_hps.policy import PatchImportancePolicy
from models.branch_cs_hps.gumbel_topk import gumbel_topk_st
from models.branch_cs_hps.decoder import CSFiLMDecoder
from models.branch_cs_hps.cs_hps_loss import CSHPSLoss
from data.datasets import RPCSSL  # 使用你的数据集类
from utils.misc import stopgrad
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image  # Ensure this is imported for image loading
import yaml
import sys

# 手动添加项目路径
sys.path.append('/root/projects/ShelfMIM')

from train.pretrain_engine import ShelfMIMPretrainModel  # 导入正确的模型类

# 1. 加载yaml配置文件
def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# 2. 加载模型和检查点
checkpoint_path = "/root/projects/ShelfMIM/outputs/finetune/rpc_samseg_gpu/iter_2250.pth"  # 确保检查点路径正确
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 加载模型
def load_model(checkpoint_path, device, cfg):
    model = ShelfMIMPretrainModel(cfg)  # 需要先加载配置，或者替换成你的具体模型
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(checkpoint.keys())  # 输出检查点中的键
    model.load_state_dict(checkpoint["state_dict"], strict=False)  # 使用 'state_dict' 键

    print(f"Model loaded from {checkpoint_path}")
    return model


# 3. 数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 修改为适合你图像的大小
    transforms.ToTensor(),
])


def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device
    return image


# 4. 推理阶段
def inference(model, image_tensor):
    model.eval()

    with torch.no_grad():
        # 生成H_b2a，作为示例：身份矩阵
        H_b2a = torch.eye(3).unsqueeze(0).to(image_tensor.device)

        # 生成 valid_mask_a 和 valid_mask_b，作为示例：全 1 掩码
        valid_mask_a = torch.ones_like(image_tensor[:, 0:1, :, :], dtype=torch.float32).to(image_tensor.device)
        valid_mask_b = torch.ones_like(image_tensor[:, 0:1, :, :], dtype=torch.float32).to(image_tensor.device)

        # 构造包含 H_b2a 和掩码的批次
        batch = {
            "img_a": image_tensor,
            "img_b": image_tensor,
            "H_b2a": H_b2a,
            "valid_mask_a": valid_mask_a,
            "valid_mask_b": valid_mask_b
        }

        fw = model(batch)

        # 获取输出：包括分割结果等
        F_a = fw["F_a"]  # ViT输出
        F_b = fw["F_b"]
        Z_a, M_a = model.slot_attn(F_a)
        Z_b, M_b = model.slot_attn(F_b)

        return Z_a, M_a, Z_b, M_b


# 5. 使用模型进行预测并保存结果
def save_result(output, save_path):
    # 根据输出处理结果
    # 你可以根据具体任务决定如何保存（比如保存为图像）
    output_image = output.squeeze(0).cpu().numpy()  # 从GPU移回到CPU
    output_image = (output_image * 255).astype('uint8')
    output_image = Image.fromarray(output_image)
    output_image.save(save_path)


# 读取目录下所有图像
def process_images_in_directory(directory_path):
    image_paths = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_paths.append(os.path.join(directory_path, filename))
    return image_paths


# 示例
image_directory = "/root/projects/ShelfMIM/dataset/archive-Retail Product Checkout Dataset/retail_product_checkout/train2019/"  # 图像目录
output_directory = "/root/projects/ShelfMIM/outputs/inference_results/"  # 预测结果保存目录
cfg_path = "/root/projects/ShelfMIM/configs/config.yaml"
cfg = load_yaml(cfg_path)  # 这里是读取配置文件

# 加载模型
model = load_model(checkpoint_path, device, cfg)

# 处理图像目录中的所有图像
image_paths = process_images_in_directory(image_directory)
for image_path in image_paths:
    print(f"Processing image: {image_path}")
    # 预处理图像
    image_tensor = preprocess_image(image_path)

    # 进行推理
    Z_a, M_a, Z_b, M_b = inference(model, image_tensor)

    # 保存结果
    output_path = os.path.join(output_directory, os.path.basename(image_path).replace(".jpg", "_prediction.jpg"))
    save_result(M_a, output_path)  # 这里假设M_a是你感兴趣的输出结果，可能是分割mask

    print(f"Inference done for {image_path}, result saved to {output_path}")
