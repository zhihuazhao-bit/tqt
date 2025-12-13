import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

class SinkhornDistance(nn.Module):
    """
    计算 Sinkhorn 距离 (正则化 Wasserstein 距离) 和 传输计划
    
    支持使用 score map 初始化文本分布 (nu)，使其反映实际的类别比例。
    """
    def __init__(self, eps=0.05, max_iter=100, reduction='none', cost_type='cos'):
        super().__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        # cost_type: 'l2' (Euclidean squared) or 'cos' (1 - cosine)
        self.cost_type = cost_type

    def forward(self, x, y, cost_matrix=None, nu_prior=None):
        """
        Args:
            x: Source distribution [B, N, C] (Image/SNE patches)
            y: Target distribution [B, M, C] (Text tokens)
            cost_matrix: [B, N, M] 代价矩阵，如果为 None 则自动计算余弦距离
            nu_prior: [B, M] 文本分布先验，如果为 None 则使用均匀分布
                      可以用 score map 的类别比例来初始化
        
        Returns:
            cost: Sinkhorn 距离
            pi: 最优传输计划 [B, N, M]
        """
        B, N, _ = x.shape
        _, M, _ = y.shape

        if cost_matrix is None:
            if self.cost_type == 'cos':
                # 使用余弦距离作为代价矩阵: C = 1 - CosSim
                x_norm = F.normalize(x, dim=2)
                y_norm = F.normalize(y, dim=2)
                cost_matrix = 1 - torch.bmm(x_norm, y_norm.transpose(1, 2))
            else:
                # 默认: 欧式距离平方代价矩阵，使用特征维度缩放降低数值范围，避免 exp(-C/eps) 下溢
                scale = x.shape[-1] ** 0.5
                x_scaled = x / scale
                y_scaled = y / scale
                cost_matrix = torch.cdist(x_scaled, y_scaled, p=2).pow(2)

        # 图像特征分布: 均匀分布 (每个 patch 质量相等)
        mu = torch.empty(B, N, dtype=x.dtype, device=x.device).fill_(1.0 / N)
        
        # 文本特征分布: 使用先验或均匀分布
        if nu_prior is not None:
            # 使用 score map 计算的类别比例作为先验
            # nu_prior: [B, M] 应该已经归一化，和为 1
            nu = nu_prior
        else:
            # 默认均匀分布
            nu = torch.empty(B, M, dtype=x.dtype, device=x.device).fill_(1.0 / M)

        # Sinkhorn 迭代
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        actual_nits = 0
        
        # Log-domain Sinkhorn for numerical stability
        thresh = 1e-1
        K = torch.exp(-cost_matrix / self.eps)

        for i in range(self.max_iter):
            u1 = u
            u = mu / (torch.bmm(K, v.unsqueeze(2)).squeeze(2) + 1e-8)
            v = nu / (torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) + 1e-8)
            err = (u - u1).abs().sum(-1).mean()
            actual_nits += 1
            if err.item() < thresh:
                break

        # 计算最优传输计划 pi [B, N, M]
        # pi = diag(u) * K * diag(v)
        pi = u.unsqueeze(2) * K * v.unsqueeze(1)
        
        # 计算 Sinkhorn 距离 (Cost)
        cost = torch.sum(pi * cost_matrix, dim=(1, 2))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi


class OTFeatureAligner(nn.Module):
    """
    OT 对齐与重心映射模块
    
    支持使用 score map 来初始化文本分布，使 OT 对齐更符合实际类别比例。
    """
    def __init__(self, dim, eps=0.05, use_score_prior=True, fuse_output=True, cost_type='cos'):
        super().__init__()
        self.sinkhorn = SinkhornDistance(eps=eps, max_iter=50, cost_type=cost_type)
        self.use_score_prior = use_score_prior
        self.fuse_output = fuse_output  # True: 返回融合后的残差输出；False: 仅返回对齐特征
        # 可选：映射后的特征融合层
        self.fusion = nn.Linear(dim * 2, dim) 
        self.norm = nn.LayerNorm(dim)
        # 仅保存一次可视化
        self._attn_saved = False

    def forward(self, source_feat, target_feat_matrix, score_map=None, attn_save_path=None, attn_class_index=1):
        """
        Args:
            source_feat: [B, HW, C] (Image or SNE)
            target_feat_matrix: [B, K, C] (Text Prompts, K=类别数)
            score_map: [B, K, H, W] 初步相似度图，用于计算类别比例作为 nu 先验
                       如果为 None 则使用均匀分布
            fuse_output: 通过构造参数控制；True 返回融合后的特征，False 返回对齐特征
        
        Returns:
            out: 对齐特征或融合特征 [B, HW, C]
            ot_loss: OT 距离损失
            pi: 传输计划
        """
        # 计算文本分布先验 (基于 score map)
        nu_prior = None
        if self.use_score_prior and score_map is not None:
            nu_prior = self._compute_nu_from_score_map(score_map)
        
        # 1. 计算 OT 距离和传输计划 pi
        ot_loss, pi = self.sinkhorn(source_feat, target_feat_matrix, nu_prior=nu_prior)

        # 1.1 如有尺度匹配，将 pi 还原为 [B, H, W, K] 并可选保存一张图的可通行权重
        if (attn_save_path is not None) and (score_map is not None) and (not self._attn_saved):
            B, K, H, W = score_map.shape
            N = source_feat.shape[1]
            if N == H * W and B > 0 and 0 <= attn_class_index < K:
                pi_map = pi.view(B, H, W, K)
                attn = pi_map[0, :, :,  ]  # 第0张图，指定类别
                attn = attn.detach().cpu()
                # 归一化到 0-1 后保存为单通道图
                attn_min, attn_max = attn.min(), attn.max()
                if (attn_max - attn_min) > 1e-8:
                    attn_norm = (attn - attn_min) / (attn_max - attn_min)
                else:
                    attn_norm = attn.clamp(min=0.0, max=1.0)
                save_image(attn_norm.unsqueeze(0), attn_save_path)
                self._attn_saved = True
        
        # 2. Barycentric Mapping (重心映射)
        # 将文本特征 target 按照 pi 的权重拉回到 source 的空间
        # [B, HW, K] x [B, K, C] -> [B, HW, C]
        pi_norm = pi / (pi.sum(dim=2, keepdim=True) + 1e-8)
        aligned_feat = torch.bmm(pi_norm, target_feat_matrix)

        if not self.fuse_output:
            return aligned_feat, ot_loss, pi

        # 3. 融合 (Residual connection)
        out = self.norm(self.fusion(torch.cat([source_feat, aligned_feat], dim=-1)))
        return out, ot_loss, pi

    def _compute_nu_from_score_map(self, score_map):
        """
        从 score map 计算文本分布先验。
        
        通过 argmax 得到每个像素的预测类别，然后统计各类别的像素数量比例。
        
        Args:
            score_map: [B, K, H, W] 每个类别的相似度图
        
        Returns:
            nu: [B, K] 归一化的类别分布（各类别像素数量比例）
        """
        B, K, H, W = score_map.shape
        
        # 1. argmax 得到每个像素的预测类别
        pred_classes = score_map.argmax(dim=1)  # [B, H, W]
        pred_flat = pred_classes.view(B, -1)     # [B, HW]
        
        # 2. 使用 one_hot + sum 统计各类别像素数量
        # one_hot: [B, HW] -> [B, HW, K]
        one_hot = F.one_hot(pred_flat, num_classes=K).float()  # [B, HW, K]
        class_counts = one_hot.sum(dim=1)  # [B, K] ← 这里是对 HW 维度求和，统计每个类别的像素数
        
        # 3. 归一化使其和为 1
        nu = class_counts / (class_counts.sum(dim=1, keepdim=True) + 1e-8)
        
        # 4. 添加平滑，避免某个类别像素数为 0
        nu = nu + 1e-6
        nu = nu / nu.sum(dim=1, keepdim=True)
        
        return nu