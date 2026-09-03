"""
层级自适应双模态融合模块 (修正版)
用于RT-DETR双模态裂纹检测

核心逻辑：
- P3层：IR缺少浅层裂纹 → VIS补充IR → re-softmax互补
- P4层：混合情况 → 双向互补 → 自适应re-softmax
- P5层：VIS缺少深层裂纹 → IR补充VIS → re-softmax互补

三层都使用re-softmax获取互补信息，只是方向不同！
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['P3ShallowFusion', 'P4MidFusion', 'P5DeepFusion']


# ============================================================================
#                     P3层融合：浅层信息，VIS补充IR
# ============================================================================
class P3ShallowFusion(nn.Module):
    """
    P3层融合模块：浅层信息处理

    问题：浅层裂纹在VIS中边缘清晰，但IR热特征弱，IR会漏掉这些小裂纹
    解决：用VIS的边缘信息补充IR，让IR也能感知浅层裂纹

    策略：
    - VIS主导引导
    - IR从VIS获取**互补**信息（re-softmax）
    - 保留VIS的边缘细节

    Args:
        c1: 输入通道数（ir_channels + vis_channels）
        c2: 输出通道数
        num_heads: 注意力头数
        window_size: 窗口大小
    """

    def __init__(self, c1, c2=None, num_heads=8, window_size=8):
        super().__init__()
        self.single_c = c1 // 2
        self.c2 = c2 if c2 is not None else c1
        self.num_heads = num_heads
        self.head_dim = self.single_c // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        # ========== VIS边缘/小目标显著性提取 ==========
        # 专门提取VIS中的浅层裂纹边缘信息
        self.vis_edge_extractor = nn.Sequential(
            # 使用小卷积核捕获细节边缘
            nn.Conv2d(self.single_c, self.single_c // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.single_c // 4),
            nn.ReLU(inplace=True),
            # 边缘增强：水平+垂直
            nn.Conv2d(self.single_c // 4, self.single_c // 4, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(self.single_c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_c // 4, self.single_c // 4, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(self.single_c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_c // 4, 1, 1),
            nn.Sigmoid()
        )

        # ========== VIS引导IR关注浅层裂纹位置 ==========
        self.vis_guide_ir = nn.Sequential(
            nn.Conv2d(self.single_c, self.single_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.single_c),
            nn.ReLU(inplace=True)
        )

        # ========== Re-Softmax互补注意力：IR从VIS获取缺失的浅层裂纹信息 ==========
        self.to_q_ir = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)  # IR作为query
        self.to_k_vis = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)  # VIS提供key
        self.to_v_vis = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)  # VIS提供value
        self.proj_ir = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)

        # ========== VIS细节保留分支 ==========
        # P3层要保留VIS的边缘细节
        self.vis_detail_branch = nn.Sequential(
            nn.Conv2d(self.single_c, self.single_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.single_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_c, self.single_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.single_c),
        )

        # ========== 融合权重 ==========
        # P3层VIS信息更可靠，权重稍高
        self.vis_weight = nn.Parameter(torch.tensor(0.55))
        self.ir_weight = nn.Parameter(torch.tensor(0.45))

        # ========== 互补强度 ==========
        self.complement_scale = nn.Parameter(torch.tensor(0.5))

        # ========== 最终融合 ==========
        self.final_fusion = nn.Sequential(
            nn.Conv2d(self.single_c * 2, self.c2, 1, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c2, self.c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: [ir_feat, vis_feat] 或 tensor
        Returns:
            fused: 融合特征
        """
        if isinstance(x, (list, tuple)):
            ir_feat, vis_feat = x[0], x[1]
        else:
            ir_feat = x[:, :self.single_c]
            vis_feat = x[:, self.single_c:]

        B, C, H, W = ir_feat.shape

        # ========== Step 1: 提取VIS中的浅层裂纹边缘信息 ==========
        vis_edge_map = self.vis_edge_extractor(vis_feat)

        # ========== Step 2: VIS引导IR关注浅层裂纹位置 ==========
        # 让IR知道哪里有浅层裂纹（即使IR自己看不到）
        ir_guided = ir_feat * (vis_edge_map * 0.5 + 0.5)
        ir_guided = self.vis_guide_ir(ir_guided)

        # ========== Step 3: Re-Softmax互补 - IR从VIS获取浅层裂纹信息 ==========
        # IR作为query（需要信息的一方）
        # VIS作为key/value（提供信息的一方）
        # re-softmax：让IR获取它与VIS**不相似**的部分（即IR缺失的浅层裂纹）
        ir_complemented = self._resoftmax_complement(
            query_feat=ir_guided,  # IR需要补充信息
            kv_feat=vis_feat,  # VIS提供浅层裂纹信息
            H=H, W=W, B=B
        )

        # ========== Step 4: VIS细节保留 ==========
        vis_enhanced = vis_feat + self.vis_detail_branch(vis_feat)

        # ========== Step 5: 加权融合 ==========
        vis_w = torch.sigmoid(self.vis_weight)
        ir_w = torch.sigmoid(self.ir_weight)

        ir_final = ir_complemented * ir_w
        vis_final = vis_enhanced * vis_w

        # ========== Step 6: 最终融合 ==========
        fused = self.final_fusion(torch.cat([ir_final, vis_final], dim=1))

        return fused

    def _resoftmax_complement(self, query_feat, kv_feat, H, W, B):
        """
        Re-Softmax互补注意力

        让query从kv中获取**互补**（不相似）的信息
        公式：attn = softmax(-QK^T / sqrt(d))
        """
        ws = self.window_size

        q = self.to_q_ir(query_feat)
        k = self.to_k_vis(kv_feat)
        v = self.to_v_vis(kv_feat)

        # Padding
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            v = F.pad(v, (0, pad_w, 0, pad_h))

        Hp, Wp = H + pad_h, W + pad_w
        nH, nW = Hp // ws, Wp // ws

        # 窗口分割
        q = self._window_partition(q, ws, B, nH, nW)
        k = self._window_partition(k, ws, B, nH, nW)
        v = self._window_partition(v, ws, B, nH, nW)

        # Re-Softmax注意力：获取互补信息
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(-attn, dim=-1)  # 关键：负号让不相似的权重更大

        out = attn @ v
        out = self._window_reverse(out, ws, Hp, Wp, B, nH, nW)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W].contiguous()

        out = self.proj_ir(out)

        # 残差连接 + 可学习的互补强度
        scale = torch.sigmoid(self.complement_scale)
        return query_feat + out * scale

    def _window_partition(self, x, ws, B, nH, nW):
        C = self.single_c
        x = x.view(B, C, nH, ws, nW, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B * nH * nW, C, ws * ws)
        x = x.view(B * nH * nW, self.num_heads, self.head_dim, ws * ws)
        x = x.permute(0, 1, 3, 2).contiguous()
        return x

    def _window_reverse(self, x, ws, Hp, Wp, B, nH, nW):
        C = self.single_c
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(B * nH * nW, C, ws * ws)
        x = x.view(B * nH * nW, C, ws, ws)
        x = x.view(B, nH, nW, C, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, Hp, Wp)
        return x


# ============================================================================
#                     P4层融合：中间层，双向互补
# ============================================================================
class P4MidFusion(nn.Module):
    """
    P4层融合模块：中间层信息处理

    问题：P4层同时包含深层和浅层裂纹信息，需要双向互补
    解决：
    - 深层裂纹区域：IR补充VIS
    - 浅层裂纹区域：VIS补充IR
    - 自适应判断每个位置的裂纹深度

    策略：双向re-softmax互补，根据深度自适应选择方向
    """

    def __init__(self, c1, c2=None, num_heads=8, window_size=8):
        super().__init__()
        self.single_c = c1 // 2
        self.c2 = c2 if c2 is not None else c1
        self.num_heads = num_heads
        self.head_dim = self.single_c // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        # ========== 双模态显著性提取 ==========
        self.ir_saliency = nn.Sequential(
            nn.Conv2d(self.single_c, self.single_c // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.single_c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_c // 4, 1, 1),
            nn.Sigmoid()
        )

        self.vis_saliency = nn.Sequential(
            nn.Conv2d(self.single_c, self.single_c // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.single_c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_c // 4, 1, 1),
            nn.Sigmoid()
        )

        # ========== 裂纹深度估计器 ==========
        # 判断当前位置是深层裂纹还是浅层裂纹
        # 深层(输出≈1): IR信息可靠，IR→VIS
        # 浅层(输出≈0): VIS信息可靠，VIS→IR
        self.depth_estimator = nn.Sequential(
            nn.Conv2d(self.single_c * 2, self.single_c // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.single_c // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_c // 2, 1, 1),
            nn.Sigmoid()
        )

        # ========== 双向引导 ==========
        self.ir_guide = nn.Sequential(
            nn.Conv2d(self.single_c, self.single_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.single_c),
            nn.ReLU(inplace=True)
        )

        self.vis_guide = nn.Sequential(
            nn.Conv2d(self.single_c, self.single_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.single_c),
            nn.ReLU(inplace=True)
        )

        # ========== 双向Re-Softmax互补注意力 ==========
        # VIS→IR (给IR补充浅层裂纹)
        self.to_q_ir = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)
        self.to_k_vis = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)
        self.to_v_vis = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)
        self.proj_ir = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)

        # IR→VIS (给VIS补充深层裂纹)
        self.to_q_vis = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)
        self.to_k_ir = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)
        self.to_v_ir = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)
        self.proj_vis = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)

        # ========== 互补强度 ==========
        self.complement_scale = nn.Parameter(torch.tensor(0.5))

        # ========== 最终融合 ==========
        self.final_fusion = nn.Sequential(
            nn.Conv2d(self.single_c * 2, self.c2, 1, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c2, self.c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            ir_feat, vis_feat = x[0], x[1]
        else:
            ir_feat = x[:, :self.single_c]
            vis_feat = x[:, self.single_c:]

        B, C, H, W = ir_feat.shape

        # ========== Step 1: 提取显著性 ==========
        ir_sal = self.ir_saliency(ir_feat)
        vis_sal = self.vis_saliency(vis_feat)

        # ========== Step 2: 估计裂纹深度 ==========
        combined = torch.cat([ir_feat, vis_feat], dim=1)
        depth_map = self.depth_estimator(combined)  # 1=深层, 0=浅层

        # ========== Step 3: 双向引导 ==========
        # 浅层区域：VIS引导IR
        ir_guided = ir_feat * (vis_sal * (1 - depth_map) + depth_map)
        ir_guided = self.ir_guide(ir_guided)

        # 深层区域：IR引导VIS
        vis_guided = vis_feat * (ir_sal * depth_map + (1 - depth_map))
        vis_guided = self.vis_guide(vis_guided)

        # ========== Step 4: 双向Re-Softmax互补 ==========
        # VIS→IR：IR从VIS获取浅层裂纹信息
        ir_complemented = self._resoftmax_vis_to_ir(ir_guided, vis_feat, H, W, B)

        # IR→VIS：VIS从IR获取深层裂纹信息
        vis_complemented = self._resoftmax_ir_to_vis(vis_guided, ir_feat, H, W, B)

        # ========== Step 5: 根据深度加权混合 ==========
        # 浅层区域更信任VIS补充后的IR
        # 深层区域更信任IR补充后的VIS
        ir_final = ir_complemented
        vis_final = vis_complemented

        # ========== Step 6: 最终融合 ==========
        fused = self.final_fusion(torch.cat([ir_final, vis_final], dim=1))

        return fused

    def _resoftmax_vis_to_ir(self, ir_feat, vis_feat, H, W, B):
        """VIS补充IR：IR从VIS获取互补的浅层裂纹信息"""
        ws = self.window_size

        q = self.to_q_ir(ir_feat)
        k = self.to_k_vis(vis_feat)
        v = self.to_v_vis(vis_feat)

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            v = F.pad(v, (0, pad_w, 0, pad_h))

        Hp, Wp = H + pad_h, W + pad_w
        nH, nW = Hp // ws, Wp // ws

        q = self._window_partition(q, ws, B, nH, nW)
        k = self._window_partition(k, ws, B, nH, nW)
        v = self._window_partition(v, ws, B, nH, nW)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(-attn, dim=-1)  # re-softmax

        out = attn @ v
        out = self._window_reverse(out, ws, Hp, Wp, B, nH, nW)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W].contiguous()

        out = self.proj_ir(out)
        scale = torch.sigmoid(self.complement_scale)
        return ir_feat + out * scale

    def _resoftmax_ir_to_vis(self, vis_feat, ir_feat, H, W, B):
        """IR补充VIS：VIS从IR获取互补的深层裂纹信息"""
        ws = self.window_size

        q = self.to_q_vis(vis_feat)
        k = self.to_k_ir(ir_feat)
        v = self.to_v_ir(ir_feat)

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            v = F.pad(v, (0, pad_w, 0, pad_h))

        Hp, Wp = H + pad_h, W + pad_w
        nH, nW = Hp // ws, Wp // ws

        q = self._window_partition(q, ws, B, nH, nW)
        k = self._window_partition(k, ws, B, nH, nW)
        v = self._window_partition(v, ws, B, nH, nW)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(-attn, dim=-1)  # re-softmax

        out = attn @ v
        out = self._window_reverse(out, ws, Hp, Wp, B, nH, nW)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W].contiguous()

        out = self.proj_vis(out)
        scale = torch.sigmoid(self.complement_scale)
        return vis_feat + out * scale

    def _window_partition(self, x, ws, B, nH, nW):
        C = self.single_c
        x = x.view(B, C, nH, ws, nW, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B * nH * nW, C, ws * ws)
        x = x.view(B * nH * nW, self.num_heads, self.head_dim, ws * ws)
        x = x.permute(0, 1, 3, 2).contiguous()
        return x

    def _window_reverse(self, x, ws, Hp, Wp, B, nH, nW):
        C = self.single_c
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(B * nH * nW, C, ws * ws)
        x = x.view(B * nH * nW, C, ws, ws)
        x = x.view(B, nH, nW, C, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, Hp, Wp)
        return x


# ============================================================================
#                     P5层融合：深层信息，IR补充VIS
# ============================================================================
class P5DeepFusion(nn.Module):
    """
    P5层融合模块：深层语义信息处理

    问题：深层裂纹在IR中热异常明显，但VIS看不清
    解决：用IR的热异常信息补充VIS，让VIS也能感知深层裂纹

    策略：
    - IR主导引导
    - VIS从IR获取**互补**信息（re-softmax）
    - 分层脱粘抑制（IR能区分裂纹和脱粘）

    Args:
        c1: 输入通道数
        c2: 输出通道数
        num_heads: 注意力头数
        window_size: 窗口大小（P5层建议用4，因为分辨率低）
    """

    def __init__(self, c1, c2=None, num_heads=8, window_size=4):
        super().__init__()
        self.single_c = c1 // 2
        self.c2 = c2 if c2 is not None else c1
        self.num_heads = num_heads
        self.head_dim = self.single_c // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        # ========== IR热异常显著性提取 ==========
        self.ir_thermal_extractor = nn.Sequential(
            nn.Conv2d(self.single_c, self.single_c // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.single_c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_c // 4, self.single_c // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.single_c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_c // 4, 1, 1),
            nn.Sigmoid()
        )

        # ========== 分层脱粘抑制 ==========
        # 裂纹检测（线性结构）
        self.crack_detector = nn.Sequential(
            nn.Conv2d(self.single_c, self.single_c // 4, (1, 3), padding=(0, 1), bias=False),
            nn.BatchNorm2d(self.single_c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_c // 4, self.single_c // 4, (3, 1), padding=(1, 0), bias=False),
            nn.BatchNorm2d(self.single_c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_c // 4, 1, 1),
            nn.Sigmoid()
        )

        # 分层脱粘检测（面状结构）
        self.delam_detector = nn.Sequential(
            nn.Conv2d(self.single_c, self.single_c // 4, 5, padding=2, bias=False),
            nn.BatchNorm2d(self.single_c // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.single_c // 4, 1, 1),
            nn.Sigmoid()
        )

        # ========== IR引导VIS关注深层裂纹位置 ==========
        self.ir_guide_vis = nn.Sequential(
            nn.Conv2d(self.single_c, self.single_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.single_c),
            nn.ReLU(inplace=True)
        )

        # ========== Re-Softmax互补注意力：VIS从IR获取深层裂纹信息 ==========
        self.to_q_vis = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)  # VIS作为query
        self.to_k_ir = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)  # IR提供key
        self.to_v_ir = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)  # IR提供value
        self.proj_vis = nn.Conv2d(self.single_c, self.single_c, 1, bias=False)

        # ========== IR语义增强分支 ==========
        self.ir_semantic_branch = nn.Sequential(
            nn.Conv2d(self.single_c, self.single_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.single_c),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.single_c, self.single_c, 1),
            nn.Sigmoid()
        )

        # ========== 融合权重（IR主导）==========
        self.ir_weight = nn.Parameter(torch.tensor(0.55))
        self.vis_weight = nn.Parameter(torch.tensor(0.45))

        # ========== 互补强度 ==========
        self.complement_scale = nn.Parameter(torch.tensor(0.5))

        # ========== 最终融合 ==========
        self.final_fusion = nn.Sequential(
            nn.Conv2d(self.single_c * 2, self.c2, 1, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.c2, self.c2, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.c2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            ir_feat, vis_feat = x[0], x[1]
        else:
            ir_feat = x[:, :self.single_c]
            vis_feat = x[:, self.single_c:]

        B, C, H, W = ir_feat.shape

        # ========== Step 1: 提取IR热异常 ==========
        ir_thermal_map = self.ir_thermal_extractor(ir_feat)

        # ========== Step 2: 分层脱粘抑制 ==========
        crack_map = self.crack_detector(ir_feat)
        delam_map = self.delam_detector(ir_feat)

        # 保留裂纹区域，抑制脱粘区域
        suppress_map = crack_map * 0.5 + 0.5  # 裂纹区域权重高
        suppress_map = suppress_map * (1 - delam_map * 0.3)  # 脱粘区域被抑制

        vis_suppressed = vis_feat * suppress_map

        # ========== Step 3: IR引导VIS关注深层裂纹位置 ==========
        vis_guided = vis_suppressed * (ir_thermal_map * 0.5 + 0.5)
        vis_guided = self.ir_guide_vis(vis_guided)

        # ========== Step 4: Re-Softmax互补 - VIS从IR获取深层裂纹信息 ==========
        vis_complemented = self._resoftmax_complement(
            query_feat=vis_guided,  # VIS需要补充信息
            kv_feat=ir_feat,  # IR提供深层裂纹信息
            H=H, W=W, B=B
        )

        # ========== Step 5: IR语义增强 ==========
        semantic_weight = self.ir_semantic_branch(ir_feat)
        ir_enhanced = ir_feat * semantic_weight

        # ========== Step 6: 加权融合 ==========
        ir_w = torch.sigmoid(self.ir_weight)
        vis_w = torch.sigmoid(self.vis_weight)

        ir_final = ir_enhanced * ir_w
        vis_final = vis_complemented * vis_w

        # ========== Step 7: 最终融合 ==========
        fused = self.final_fusion(torch.cat([ir_final, vis_final], dim=1))

        return fused

    def _resoftmax_complement(self, query_feat, kv_feat, H, W, B):
        """Re-Softmax互补：VIS从IR获取互补的深层裂纹信息"""
        ws = self.window_size

        q = self.to_q_vis(query_feat)
        k = self.to_k_ir(kv_feat)
        v = self.to_v_ir(kv_feat)

        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            q = F.pad(q, (0, pad_w, 0, pad_h))
            k = F.pad(k, (0, pad_w, 0, pad_h))
            v = F.pad(v, (0, pad_w, 0, pad_h))

        Hp, Wp = H + pad_h, W + pad_w
        nH, nW = Hp // ws, Wp // ws

        q = self._window_partition(q, ws, B, nH, nW)
        k = self._window_partition(k, ws, B, nH, nW)
        v = self._window_partition(v, ws, B, nH, nW)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(-attn, dim=-1)  # re-softmax：获取互补信息

        out = attn @ v
        out = self._window_reverse(out, ws, Hp, Wp, B, nH, nW)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W].contiguous()

        out = self.proj_vis(out)
        scale = torch.sigmoid(self.complement_scale)
        return query_feat + out * scale

    def _window_partition(self, x, ws, B, nH, nW):
        C = self.single_c
        x = x.view(B, C, nH, ws, nW, ws)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
        x = x.view(B * nH * nW, C, ws * ws)
        x = x.view(B * nH * nW, self.num_heads, self.head_dim, ws * ws)
        x = x.permute(0, 1, 3, 2).contiguous()
        return x

    def _window_reverse(self, x, ws, Hp, Wp, B, nH, nW):
        C = self.single_c
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.view(B * nH * nW, C, ws * ws)
        x = x.view(B * nH * nW, C, ws, ws)
        x = x.view(B, nH, nW, C, ws, ws)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.view(B, C, Hp, Wp)
        return x