import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import Block
from ._base import BaseDistiller
from .registry import register_distiller
from .utils import GAP1d, get_module_dict, init_weights, is_cnn_model, PatchMerging, SepConv, set_module_dict, \
    TokenFilter, TokenFnContext, kd_loss, spearman_correlation
from .dist import dist_loss


def cff_loss(self, logits_student, logits_teacher, logits_cff, target_mask, decay_ratio, delta=1.5):
    tau = 4.0

    logits_cff = (1 - decay_ratio) * logits_cff + decay_ratio * logits_student

    log_pred_cff = F.log_softmax(logits_cff / tau, dim=1)
    pred_teacher = F.softmax(logits_teacher / tau, dim=1)
    pred = (pred_teacher + target_mask) ** delta - target_mask
    l_cff = torch.sum(- pred * log_pred_cff, dim=-1).mean()

    # l_dist = dist_loss(logits_student, logits_teacher)
    l_spear = 1 - spearman_correlation(logits_teacher, logits_student)

    loss = l_cff + l_spear

    return loss


def get_decay_ratio(self, epoch):
    x = epoch / self.args.epochs
    if self.args.cff_decay_func == 'linear':
        return 1 - x
    elif self.args.cff_decay_func == 'x2':
        return (1 - x) ** 2
    elif self.args.cff_decay_func == 'cos':
        return math.cos(math.pi * 0.5 * x)
    else:
        raise NotImplementedError(self.args.cff_decay_func)


class MHAFusion(nn.Module):
    def __init__(self, d_model, num_classes, num_heads=6):
        super(MHAFusion, self).__init__()
        self.out_layer = nn.Linear(in_features=d_model, out_features=num_classes)

        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        self.norm_layer = nn.LayerNorm(d_model)
        self.final_norm = nn.LayerNorm(d_model)

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

    def forward(self, feat_student, feat_teacher):
        feat_teacher = feat_teacher[:, 1:, :]  # 去除CLS token

        B, N, E = feat_teacher.size()

        # 调整feat_student形状,对齐feat_teacher
        feat_student = feat_student.reshape(B, E, -1).permute(0, 2, 1)

        # 线性投影
        query = self.query_proj(feat_student)
        key = self.key_proj(feat_teacher)
        value = self.value_proj(feat_student)

        # 层归一化
        query = self.norm_layer(query)
        key = self.norm_layer(key)
        value = self.norm_layer(value)

        # 分割到多头
        query = query.view(B, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        key = key.view(B, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3).transpose(-2, -1)
        value = value.view(B, -1, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        # 计算相似度
        similarity = torch.matmul(query, key) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        # 计算注意力权重
        attention_weights = F.softmax(similarity, dim=-1)
        # 加权融合
        weighted_feat = torch.matmul(attention_weights, value)
        # 多头汇聚
        weighted_feat = weighted_feat.permute(0, 2, 1, 3).reshape(B, -1, self.d_model)

        # 残差连接
        weighted_feat += feat_student

        cumulative_logits = self.final_norm(weighted_feat)
        # 全局池化
        cumulative_logits = GAP1d()(cumulative_logits)
        # 输出层
        logits_cff = self.out_layer(cumulative_logits)

        return logits_cff


@register_distiller
class CFF(BaseDistiller):
    requires_feat = True

    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(CFF, self).__init__(student, teacher, criterion, args)

        _, feature_dim_t = self.teacher.stage_info(-1)
        _, feature_dim_s = self.student.stage_info(-1)

        # 对齐 teacher_logits 的投影头
        self.projector = nn.Sequential(
            nn.Conv2d(in_channels=feature_dim_s, out_channels=feature_dim_t, kernel_size=1),  # 1x1卷积用于通道转换
            nn.Upsample(size=(14, 14), mode='bilinear', align_corners=False)  # align token，线性插值上采样
        )
        self.attention = MHAFusion(feature_dim_t, args.num_classes)

        self.projector.apply(init_weights)
        # print(self.projector)  # for debug

    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)

        logits_student, feat_student = self.student(image, requires_feat=True)

        num_classes = logits_student.size(-1)
        if len(label.shape) != 1:  # label smoothing
            target_mask = F.one_hot(label.argmax(-1), num_classes)
        else:
            target_mask = F.one_hot(label, num_classes)

        # Transform
        feat_stu = self.projector(feat_student[-2])

        # CFF 权重衰退率
        decay_ratio = get_decay_ratio(self, epoch=kwargs['epoch'])

        logits_cff = self.attention(feat_stu, feat_teacher[-2])
        loss_cff = self.args.cff_loss_weight * cff_loss(self, logits_student, logits_teacher, logits_cff, target_mask,
                                                        decay_ratio)

        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)

        losses_dict = {
            "loss_gt": loss_gt,
            "loss_cff": loss_cff
        }
        return logits_student, losses_dict
