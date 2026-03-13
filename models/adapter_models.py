# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class LinearAdapterWithLayerNorm(nn.Module):
#     def __init__(self, hidden_dim, projection_dim):
#         super(LinearAdapterWithLayerNorm, self).__init__()
#         self.linear = nn.Linear(hidden_dim, projection_dim)
#         self.layer_norm = nn.LayerNorm(projection_dim)
    
#     def forward(self, x):
#         # Input first passes through the linear layer
#         x = self.linear(x)
#         # Then through layer normalization
#         x = self.layer_norm(x)
#         return x



import torch
import torch.nn as nn
import torch.nn.functional as F

class StyleModulatedAdapter(nn.Module):
    def __init__(self, content_dim=128, style_dim=512, projection_dim=4096, use_style=True):
        """
        content_dim: 字形特征 + 坐标特征 + 序列特征 (128)
        style_dim: 风格特征维度 (512)
        use_style: 【消融实验开关】设为 False 时，退化为基础的 Linear + LayerNorm
        """
        super(StyleModulatedAdapter, self).__init__()
        self.use_style = use_style
        
        # --- 1. 内容主分支 (始终保留) ---
        self.linear = nn.Linear(content_dim, projection_dim)
        
        # 【消融细节】：
        # 如果不开风格，我们使用带仿射参数的普通 LayerNorm (等价于您的Baseline)。
        # 如果开启风格，关闭 LayerNorm 自带的仿射参数，交由风格特征去生成 scale 和 shift。
        self.layer_norm = nn.LayerNorm(
            projection_dim, 
            elementwise_affine=(not self.use_style)
        ) 
        
        # --- 2. 风格调制分支 (受开关控制) ---
        if self.use_style:
            self.style_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(style_dim, projection_dim * 2)
            )
            # 零初始化策略：确保第一步训练时不破坏网络
            nn.init.zeros_(self.style_mlp[1].weight)
            nn.init.zeros_(self.style_mlp[1].bias)

    def forward(self, content_feat, style_feat=None):
        """
        content_feat: [B, Seq_Len, content_dim] 
        style_feat:   [B, style_dim] (关闭风格时可传入 None)
        """
        # 1. 骨架：映射内容特征并归一化
        x = self.linear(content_feat)      
        x = self.layer_norm(x)             

        # 2. 【消融逻辑】：如果不使用风格，直接返回纯净的内容骨架
        if not self.use_style or style_feat is None:
            return x

        # 3. 染料：计算调制参数
        if style_feat.dim() == 2:
            style_feat = style_feat.unsqueeze(1) # [B, 1, 512]
            
        style_params = self.style_mlp(style_feat) # [B, 1, 4096 * 2]
        scale, shift = style_params.chunk(2, dim=-1) # 各自是 [B, 1, 4096]

        # 4. 渲染：执行 AdaLN 调制
        x = x * (1 + scale) + shift
        
        return x