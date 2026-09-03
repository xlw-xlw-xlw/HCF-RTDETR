"""
CrackCFT: 针对裂纹检测优化的跨模态融合Transformer (修复版)

改进点:
1. 条形池化 - 保留裂纹的线性结构
2. 显著性引导 - 让Attention关注裂纹区域
3. 模态自适应权重 - VIS/IR信息不对等时自适应调整
4. 层级差异化配置 - P2/P3/P4/P5不同配置

使用方法:
在yaml中注册后使用:
- [[vis_idx, ir_idx], 1, CrackCFT_P2, [64]]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

__all__ = [
    'CrackCFT',
    'CrackCFT_P2',
    'CrackCFT_P3',
    'CrackCFT_P4',
    'CrackCFT_P5',
    'Add2'
]


# ============================================================================
#                         Transformer Block
# ============================================================================
class CrackTransformerBlock(nn.Module):
    """Transformer Block with parameter validation"""

    def __init__(self, d_model, n_head=8, mlp_ratio=4, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        # ========== 参数验证和自动调整 ==========
        if n_head > d_model:
            n_head = d_model
        while d_model % n_head != 0 and n_head > 1:
            n_head -= 1
        if n_head == 0:
            n_head = 1

        self.n_head = n_head
        self.d_k = d_model // n_head

        # 确保 d_k 至少为 1
        if self.d_k == 0:
            self.d_k = 1
            self.n_head = d_model

        self.scale = self.d_k ** -0.5

        # Layer Norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        # Multi-head Self-Attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)

        # MLP (Feed Forward)
        hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(resid_pdrop),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(resid_pdrop)
        )

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: [B, N, C] - token序列
        Returns:
            out: [B, N, C]
        """
        bs, n, c = x.shape

        # ========== Self-Attention ==========
        x_ln = self.ln1(x)

        # Q, K, V projection
        q = self.q_proj(x_ln).view(bs, n, self.n_head, self.d_k).permute(0, 2, 1, 3)  # [B, h, N, d_k]
        k = self.k_proj(x_ln).view(bs, n, self.n_head, self.d_k).permute(0, 2, 1, 3)  # [B, h, N, d_k]
        v = self.v_proj(x_ln).view(bs, n, self.n_head, self.d_k).permute(0, 2, 1, 3)  # [B, h, N, d_k]

        # Attention: softmax(QK^T / sqrt(d_k)) * V
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, h, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Output
        out = (attn @ v).permute(0, 2, 1, 3).contiguous().view(bs, n, c)  # [B, N, C]
        out = self.resid_drop(self.out_proj(out))

        x = x + out  # Residual

        # ========== MLP ==========
        x = x + self.mlp(self.ln2(x))

        return x


# ============================================================================
#                         CrackCFT 主模块
# ============================================================================
class CrackCFT(nn.Module):
    """
    针对裂纹检测优化的跨模态融合Transformer

    Args:
        c1: 输入通道数 (实际是 d_model，因为yaml传入的是通道数)
        n_head: 注意力头数
        n_layer: Transformer层数
        pool_h: 池化高度
        pool_w: 池化宽度
        use_strip_pool: 是否使用条形池化（保留线性结构）
        use_saliency_guide: 是否使用显著性引导
        use_modality_weight: 是否使用模态自适应权重
        mlp_ratio: MLP扩展比例
        attn_pdrop: Attention dropout
        resid_pdrop: Residual dropout
    """

    def __init__(self, c1, n_head=8, n_layer=2,
                 pool_h=8, pool_w=8,
                 use_strip_pool=True,
                 use_saliency_guide=True,
                 use_modality_weight=True,
                 mlp_ratio=4,
                 attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()

        # ========== 参数验证 ==========
        assert c1 > 0, f"c1 must be positive, got {c1}"

        # 自动调整 n_head，确保能整除 c1
        if n_head > c1:
            n_head = c1
        while c1 % n_head != 0 and n_head > 1:
            n_head -= 1
        if n_head == 0:
            n_head = 1

        # c1 是yaml传入的通道数，作为d_model
        self.d_model = c1
        self.n_head = n_head
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.use_strip_pool = use_strip_pool
        self.use_saliency_guide = use_saliency_guide
        self.use_modality_weight = use_modality_weight

        # ========== 1. 池化层 ==========
        if use_strip_pool:
            # 条形池化：保留水平和垂直方向的线性结构
            self.pool_h_strip = nn.AdaptiveAvgPool2d((pool_h, 1))  # 垂直条 [B,C,H,1]
            self.pool_w_strip = nn.AdaptiveAvgPool2d((1, pool_w))  # 水平条 [B,C,1,W]

            # 确保 pool_h//2 和 pool_w//2 至少为 1
            pool_h_half = max(pool_h // 2, 1)
            pool_w_half = max(pool_w // 2, 1)
            self.pool_square = nn.AdaptiveAvgPool2d((pool_h_half, pool_w_half))  # 方形
            self.pool_h_half = pool_h_half
            self.pool_w_half = pool_w_half

            # 融合三种池化的输出
            self.pool_fusion = nn.Sequential(
                nn.Conv2d(c1 * 3, c1, 1, bias=False),
                nn.BatchNorm2d(c1),
                nn.ReLU(inplace=True)
            )

            # 计算token数量
            self.n_tokens = pool_h + pool_w + pool_h_half * pool_w_half
        else:
            self.avgpool = nn.AdaptiveAvgPool2d((pool_h, pool_w))
            self.n_tokens = pool_h * pool_w

        # ========== 2. 显著性引导 ==========
        if use_saliency_guide:
            # 确保通道数至少为 1
            hidden_ch = max(c1 // 4, 1)

            # VIS显著性提取（边缘/裂纹区域）
            self.vis_saliency = nn.Sequential(
                nn.Conv2d(c1, hidden_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, 1, 1),
                nn.Sigmoid()
            )
            # IR显著性提取（热异常区域）
            self.ir_saliency = nn.Sequential(
                nn.Conv2d(c1, hidden_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(hidden_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_ch, 1, 1),
                nn.Sigmoid()
            )

        # ========== 3. 模态自适应权重 ==========
        if use_modality_weight:
            # 确保通道数至少为 1
            gate_hidden = max(c1 // 2, 1)

            self.modality_gate = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c1 * 2, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, 2),
                nn.Softmax(dim=1)
            )

        # ========== 4. 位置编码 ==========
        # VIS和IR各有n_tokens个token，共2*n_tokens
        self.pos_emb = nn.Parameter(torch.zeros(1, 2 * self.n_tokens, c1))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        # ========== 5. Transformer Blocks ==========
        self.trans_blocks = nn.ModuleList([
            CrackTransformerBlock(c1, n_head, mlp_ratio, attn_pdrop, resid_pdrop)
            for _ in range(n_layer)
        ])

        # ========== 6. 输出层 ==========
        self.ln_out = nn.LayerNorm(c1)
        self.drop = nn.Dropout(resid_pdrop)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _strip_pool(self, x):
        """
        条形池化：保留裂纹的线性结构

        Args:
            x: [B, C, H, W]
        Returns:
            x_pool: [B, C, n_tokens]
        """
        bs, c, h, w = x.shape

        # 垂直条: [B, C, pool_h, 1] → [B, C, pool_h]
        x_h = self.pool_h_strip(x).squeeze(-1)

        # 水平条: [B, C, 1, pool_w] → [B, C, pool_w]
        x_w = self.pool_w_strip(x).squeeze(-2)

        # 方形: [B, C, pool_h//2, pool_w//2] → [B, C, n_square]
        x_s = self.pool_square(x)
        x_s = x_s.view(bs, c, -1)

        # 拼接: [B, C, pool_h + pool_w + n_square]
        x_pool = torch.cat([x_h, x_w, x_s], dim=2)

        return x_pool

    def _strip_unpool(self, tokens, h, w):
        """
        条形池化的逆操作：从tokens恢复到特征图

        Args:
            tokens: [B, n_tokens, C]
            h, w: 目标高宽
        Returns:
            feat: [B, C, H, W]
        """
        bs = tokens.shape[0]
        c = self.d_model

        n_h = self.pool_h
        n_w = self.pool_w
        n_s = self.pool_h_half * self.pool_w_half

        # 分割三部分
        t_h = tokens[:, :n_h, :]  # [B, pool_h, C]
        t_w = tokens[:, n_h:n_h + n_w, :]  # [B, pool_w, C]
        t_s = tokens[:, n_h + n_w:, :]  # [B, n_s, C]

        # 垂直条: [B, pool_h, C] → [B, C, pool_h, 1] → [B, C, H, W]
        f_h = t_h.permute(0, 2, 1).unsqueeze(-1)
        f_h = F.interpolate(f_h, size=(h, w), mode='bilinear', align_corners=False)

        # 水平条: [B, pool_w, C] → [B, C, 1, pool_w] → [B, C, H, W]
        f_w = t_w.permute(0, 2, 1).unsqueeze(-2)
        f_w = F.interpolate(f_w, size=(h, w), mode='bilinear', align_corners=False)

        # 方形: [B, n_s, C] → [B, C, pool_h//2, pool_w//2] → [B, C, H, W]
        f_s = t_s.permute(0, 2, 1).view(bs, c, self.pool_h_half, self.pool_w_half)
        f_s = F.interpolate(f_s, size=(h, w), mode='bilinear', align_corners=False)

        # 拼接并融合: [B, 3C, H, W] → [B, C, H, W]
        f_cat = torch.cat([f_h, f_w, f_s], dim=1)
        f_out = self.pool_fusion(f_cat)

        return f_out

    def forward(self, x):
        """
        Args:
            x: [vis_feat, ir_feat] 或 拼接的tensor
               vis_feat: [B, C, H, W]
               ir_feat: [B, C, H, W]
        Returns:
            (vis_out, ir_out): 各为 [B, C, H, W]
        """
        # 处理输入
        if isinstance(x, (list, tuple)):
            vis_feat, ir_feat = x[0], x[1]
        else:
            c = x.shape[1] // 2
            vis_feat = x[:, :c]
            ir_feat = x[:, c:]

        bs, c, h, w = vis_feat.shape

        # ========== 1. 显著性引导（可选）==========
        if self.use_saliency_guide:
            vis_sal = self.vis_saliency(vis_feat)  # [B, 1, H, W]
            ir_sal = self.ir_saliency(ir_feat)  # [B, 1, H, W]

            # 增强显著区域，但不完全抑制背景
            vis_enhanced = vis_feat * (vis_sal * 0.5 + 0.5)
            ir_enhanced = ir_feat * (ir_sal * 0.5 + 0.5)
        else:
            vis_enhanced = vis_feat
            ir_enhanced = ir_feat

        # ========== 2. 池化 ==========
        if self.use_strip_pool:
            vis_pool = self._strip_pool(vis_enhanced)  # [B, C, n_tokens]
            ir_pool = self._strip_pool(ir_enhanced)  # [B, C, n_tokens]
        else:
            vis_pool = self.avgpool(vis_enhanced).view(bs, c, -1)  # [B, C, pool_h*pool_w]
            ir_pool = self.avgpool(ir_enhanced).view(bs, c, -1)

        # ========== 3. 拼接 + 位置编码 ==========
        # [B, C, 2*n_tokens] → [B, 2*n_tokens, C]
        tokens = torch.cat([vis_pool, ir_pool], dim=2).permute(0, 2, 1)
        tokens = self.drop(tokens + self.pos_emb)

        # ========== 4. Transformer ==========
        for block in self.trans_blocks:
            tokens = block(tokens)

        tokens = self.ln_out(tokens)

        # ========== 5. 分割VIS和IR ==========
        vis_tokens = tokens[:, :self.n_tokens, :]  # [B, n_tokens, C]
        ir_tokens = tokens[:, self.n_tokens:, :]  # [B, n_tokens, C]

        # ========== 6. 上采样恢复 ==========
        if self.use_strip_pool:
            vis_out = self._strip_unpool(vis_tokens, h, w)
            ir_out = self._strip_unpool(ir_tokens, h, w)
        else:
            vis_out = vis_tokens.permute(0, 2, 1).view(bs, c, self.pool_h, self.pool_w)
            vis_out = F.interpolate(vis_out, size=(h, w), mode='bilinear', align_corners=False)
            ir_out = ir_tokens.permute(0, 2, 1).view(bs, c, self.pool_h, self.pool_w)
            ir_out = F.interpolate(ir_out, size=(h, w), mode='bilinear', align_corners=False)

        # ========== 7. 模态自适应权重（可选）==========
        if self.use_modality_weight:
            # 计算每个模态的全局特征
            vis_global = F.adaptive_avg_pool2d(vis_feat, 1)  # [B, C, 1, 1]
            ir_global = F.adaptive_avg_pool2d(ir_feat, 1)  # [B, C, 1, 1]
            combined = torch.cat([vis_global, ir_global], dim=1)  # [B, 2C, 1, 1]

            # 计算权重
            weights = self.modality_gate(combined)  # [B, 2]
            w_vis = weights[:, 0:1].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]
            w_ir = weights[:, 1:2].unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1, 1]

            # 应用权重（缩放到合理范围 [0.5, 1.5]）
            vis_out = vis_out * (w_vis + 0.5)
            ir_out = ir_out * (w_ir + 0.5)

        return vis_out, ir_out


# ============================================================================
#                    不同层级的CrackCFT配置
# ============================================================================

class CrackCFT_P2(CrackCFT):
    """
    P2层: 高分辨率，浅层特征，细节丰富
    - 使用条形池化保留细节
    - 使用显著性引导关注裂纹
    - 不使用模态权重（浅层两者差异不大）
    """

    def __init__(self, c1):
        # 确保 n_head 能整除 c1
        n_head = 4
        while c1 % n_head != 0 and n_head > 1:
            n_head -= 1
        if n_head == 0:
            n_head = 1

        super().__init__(
            c1=c1,
            n_head=n_head,
            n_layer=1,
            pool_h=16, pool_w=16,
            use_strip_pool=True,
            use_saliency_guide=True,
            use_modality_weight=False,
            mlp_ratio=4,
            attn_pdrop=0.1,
            resid_pdrop=0.1
        )


class CrackCFT_P3(CrackCFT):
    """
    P3层: 中等分辨率，浅层裂纹明显
    - VIS边缘清晰，IR热特征弱
    - 开始需要模态权重
    """

    def __init__(self, c1):
        # 确保 n_head 能整除 c1
        n_head = 8
        while c1 % n_head != 0 and n_head > 1:
            n_head -= 1
        if n_head == 0:
            n_head = 1

        super().__init__(
            c1=c1,
            n_head=n_head,
            n_layer=2,
            pool_h=12, pool_w=12,
            use_strip_pool=True,
            use_saliency_guide=True,
            use_modality_weight=True,
            mlp_ratio=4,
            attn_pdrop=0.1,
            resid_pdrop=0.1
        )


class CrackCFT_P4(CrackCFT):
    """
    P4层: 较低分辨率，深浅裂纹混合
    - 需要综合考虑两种模态
    """

    def __init__(self, c1):
        # 确保 n_head 能整除 c1
        n_head = 8
        while c1 % n_head != 0 and n_head > 1:
            n_head -= 1
        if n_head == 0:
            n_head = 1

        super().__init__(
            c1=c1,
            n_head=n_head,
            n_layer=2,
            pool_h=8, pool_w=8,
            use_strip_pool=True,
            use_saliency_guide=True,
            use_modality_weight=True,
            mlp_ratio=4,
            attn_pdrop=0.1,
            resid_pdrop=0.1
        )


class CrackCFT_P5(CrackCFT):
    """
    P5层: 低分辨率，深层语义特征
    - IR热异常明显，VIS看不清
    - 不需要条形池化（语义层）
    - 不需要显著性引导（语义层）
    """

    def __init__(self, c1):
        # 确保 n_head 能整除 c1
        n_head = 8
        while c1 % n_head != 0 and n_head > 1:
            n_head -= 1
        if n_head == 0:
            n_head = 1

        super().__init__(
            c1=c1,
            n_head=n_head,
            n_layer=2,
            pool_h=4, pool_w=4,
            use_strip_pool=False,
            use_saliency_guide=False,
            use_modality_weight=True,
            mlp_ratio=4,
            attn_pdrop=0.1,
            resid_pdrop=0.1
        )


# ============================================================================
#                         Add2 模块（残差连接）
# ============================================================================

class Add2(nn.Module):
    """
    将CFT输出加回原特征

    用法:
        - [[vis_feat_idx, cft_output_idx], 1, Add2, [channels, 0]]  # VIS + CFT[0]
        - [[ir_feat_idx, cft_output_idx], 1, Add2, [channels, 1]]   # IR + CFT[1]
    """

    def __init__(self, c1, index):
        super().__init__()
        self.index = index  # 0 for VIS, 1 for IR

    def forward(self, x):
        """
        Args:
            x: [src_feat, cft_output]
               src_feat: [B, C, H, W] - 原始特征
               cft_output: (vis_out, ir_out) - CFT输出的元组
        Returns:
            out: [B, C, H, W] - src_feat + cft_output[index]
        """
        src, cft_out = x[0], x[1]

        # CFT输出是元组 (vis_out, ir_out)
        if isinstance(cft_out, (list, tuple)):
            trans_part = cft_out[self.index]
        else:
            # 如果不是元组，假设是拼接的tensor
            c = cft_out.shape[1] // 2
            trans_part = cft_out[:, :c] if self.index == 0 else cft_out[:, c:]

        # 尺寸对齐
        if src.shape[2:] != trans_part.shape[2:]:
            trans_part = F.interpolate(
                trans_part,
                size=src.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        return src + trans_part


# ============================================================================
#                         测试代码
# ============================================================================

if __name__ == '__main__':
    # 测试不同层级的CrackCFT
    print("Testing CrackCFT modules...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # P2测试
    print("\n[P2] Testing CrackCFT_P2 with c1=64...")
    model_p2 = CrackCFT_P2(c1=64).to(device)
    vis_p2 = torch.randn(2, 64, 160, 160).to(device)
    ir_p2 = torch.randn(2, 64, 160, 160).to(device)
    vis_out, ir_out = model_p2([vis_p2, ir_p2])
    print(f"  Input:  VIS {vis_p2.shape}, IR {ir_p2.shape}")
    print(f"  Output: VIS {vis_out.shape}, IR {ir_out.shape}")

    # P3测试
    print("\n[P3] Testing CrackCFT_P3 with c1=128...")
    model_p3 = CrackCFT_P3(c1=128).to(device)
    vis_p3 = torch.randn(2, 128, 80, 80).to(device)
    ir_p3 = torch.randn(2, 128, 80, 80).to(device)
    vis_out, ir_out = model_p3([vis_p3, ir_p3])
    print(f"  Input:  VIS {vis_p3.shape}, IR {ir_p3.shape}")
    print(f"  Output: VIS {vis_out.shape}, IR {ir_out.shape}")

    # P4测试
    print("\n[P4] Testing CrackCFT_P4 with c1=256...")
    model_p4 = CrackCFT_P4(c1=256).to(device)
    vis_p4 = torch.randn(2, 256, 40, 40).to(device)
    ir_p4 = torch.randn(2, 256, 40, 40).to(device)
    vis_out, ir_out = model_p4([vis_p4, ir_p4])
    print(f"  Input:  VIS {vis_p4.shape}, IR {ir_p4.shape}")
    print(f"  Output: VIS {vis_out.shape}, IR {ir_out.shape}")

    # P5测试
    print("\n[P5] Testing CrackCFT_P5 with c1=512...")
    model_p5 = CrackCFT_P5(c1=512).to(device)
    vis_p5 = torch.randn(2, 512, 20, 20).to(device)
    ir_p5 = torch.randn(2, 512, 20, 20).to(device)
    vis_out, ir_out = model_p5([vis_p5, ir_p5])
    print(f"  Input:  VIS {vis_p5.shape}, IR {ir_p5.shape}")
    print(f"  Output: VIS {vis_out.shape}, IR {ir_out.shape}")

    # 边界情况测试
    print("\n[Edge Case] Testing with small channel count c1=32...")
    model_small = CrackCFT_P2(c1=32).to(device)
    vis_small = torch.randn(2, 32, 80, 80).to(device)
    ir_small = torch.randn(2, 32, 80, 80).to(device)
    vis_out, ir_out = model_small([vis_small, ir_small])
    print(f"  Input:  VIS {vis_small.shape}, IR {ir_small.shape}")
    print(f"  Output: VIS {vis_out.shape}, IR {ir_out.shape}")

    # Add2测试
    print("\n[Add2] Testing Add2...")
    add2_vis = Add2(c1=128, index=0)
    add2_ir = Add2(c1=128, index=1)

    src = torch.randn(2, 128, 80, 80).to(device)
    cft_output = (torch.randn(2, 128, 80, 80).to(device),
                  torch.randn(2, 128, 80, 80).to(device))

    out_vis = add2_vis([src, cft_output])
    out_ir = add2_ir([src, cft_output])
    print(f"  Add2 VIS output: {out_vis.shape}")
    print(f"  Add2 IR output: {out_ir.shape}")

    # 参数量统计
    print("\n" + "=" * 50)
    print("Parameter Count:")
    print("=" * 50)
    for name, model in [('CrackCFT_P2(64)', CrackCFT_P2(64)),
                        ('CrackCFT_P3(128)', CrackCFT_P3(128)),
                        ('CrackCFT_P4(256)', CrackCFT_P4(256)),
                        ('CrackCFT_P5(512)', CrackCFT_P5(512))]:
        params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {params:,} ({params / 1e6:.2f}M)")

    print("\n✓ All tests passed!")