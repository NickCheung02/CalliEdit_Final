import torch
import torch.nn as nn
import torch.nn.functional as F

# ✅ 路径已更新为重命名后的 Style_ocr_recog
from Style_ocr_recog.RecMv1_enhance import MobileNetV1Enhance

class FontStyleEncoder(nn.Module):
    def __init__(self, style_dim=512, use_style=True):
        """
        style_dim: 提取出的风格特征的最终维度 (默认为 512)
        use_style: 【消融实验开关】
        """
        super().__init__()
        self.use_style = use_style
        self.style_dim = style_dim
        
        if self.use_style:
            # 实例化 PP-OCRv3 的 MobileNetV1Enhance 骨干网络
            self.backbone = MobileNetV1Enhance(in_channels=3, scale=0.5)
            
            # 冻结 Backbone 权重
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, ref_image):
        """
        ref_image: [B, 3, H, W] 书法参考图
        """
        B = ref_image.shape[0]
        device = ref_image.device
        dtype = ref_image.dtype
        
        # 【消融开关】关闭时直接返回全 0 张量
        if not self.use_style:
            return torch.zeros((B, self.style_dim), device=device, dtype=dtype)
            
        feat = self.backbone(ref_image)
        style_feat = F.adaptive_avg_pool2d(feat, (1, 1))
        style_feat = style_feat.view(B, -1)
        
        return style_feat

    def load_pretrained_weights(self, weight_path="./Style_ocr_weights/ppv3_rec.pth"):
        """
        ✅ 默认路径已指向您的 Style_ocr_weights 文件夹
        """
        if not self.use_style:
            return
            
        state_dict = torch.load(weight_path, map_location='cpu')
        backbone_dict = {}
        for k, v in state_dict.items():
            if k.startswith('backbone.'):
                backbone_dict[k.replace('backbone.', '')] = v
                
        self.backbone.load_state_dict(backbone_dict, strict=True)
        print(f"✅ 成功加载 PP-OCRv3 Backbone 预训练权重！")