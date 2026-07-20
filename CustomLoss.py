# Ultralytics  AGPL-3.0 License - https://ultralytics.com/license
# Modified for Ablation Study - Parameters read from model.args
#
# FIXES APPLIED (vs previous version):
#   [FIX-1] BboxLoss._compute_weights: areas converted grid-units -> PIXELS
#           using each anchor's own stride. small_obj_px is a clean pixel-area
#           threshold (area < small_obj_px^2 at training resolution).
#   [FIX-2] BboxLoss.forward: clip-rate tracking, printed per epoch.
#   [FIX-3] BboxLoss._get_dynamic_alpha: per-epoch log with alpha + clip rates.
#   [FIX-4] _compute_center_loss: pixel space + scale-free error.
#   [FIX-5] v8SegmentationLoss: fg_mask.sum() parentheses crash fixed.
#
# NEW IN THIS VERSION (Round 9 revision -- informed by v6/r0/r7/r8 results):
#   [NEW-1] WEIGHT RENORMALIZATION (weight_renorm, default 1).
#           The combined weight alpha*area_w + (1-alpha)*score_w is rescaled so
#           its sum equals the score-weight sum. Section A now redistributes
#           loss across samples WITHOUT changing the effective box/dfl gain
#           relative to cls. This is the leading suspect for the r7_stack
#           collapse (-4.3 mAP): un-renormalized weights shifted total loss
#           magnitude, and stacked components compounded the shift into the
#           clip caps. With alpha=0 the renorm is exactly identity, so all
#           "sections off" anchor runs reproduce bit-for-bit.
#           Set weight_renorm=0 to reproduce pre-NEW behavior.
#   [NEW-2] FIXED-REFERENCE AREA WEIGHTING (area_mode='fixed', default).
#           Replaces the batch-relative 1/area normalized by batch max (an
#           object's weight depended on the smallest box sharing its batch --
#           pure noise). New: area_weight = (area_ref_px^2 / area_px)^area_gamma,
#           clamped to area_w_cap. area_gamma=0.5 is the sqrt variant that
#           r8_area_sqrt suggested; deterministic per object, reproducible.
#           area_mode='legacy' restores the old batch-relative scheme.
#   [NEW-3] dfl_small_boost (default 1.0 = off). Multiplies ONLY the DFL term
#           for small objects (area < small_obj_px^2), independent of alpha.
#           Targets the diagnosed deficit (AR50_small ~0.96 vs AP50-95_small
#           ~0.51 = edge-precision, DFL's job). The boost is applied BEFORE the
#           per-sample cap, so tight Section-C caps can absorb it -- run with
#           clips off or watch the clip-rate log. This is the mechanism the
#           r0a2_dflboost run was supposed to test (it silently no-op'd:
#           identical metrics to r0_default2 confirmed the parameter was never
#           read).
#           [NEW-3b] IoU-GATED DFL BOOST. When dfl_iou_gated=1, the flat
#           dfl_small_boost is modulated by (1-IoU) per anchor: only poorly-
#           localized small objects get the full boost. Well-localized small
#           objects (IoU~0.9) get nearly no boost. This focuses DFL sharpening
#           exactly where the model still struggles, rather than wasting capacity
#           on already-tight boxes. Off-switch: dfl_iou_gated=0 (default)
#           preserves legacy flat-boost behavior.
#   [NEW-4] VARIFOCAL CLASSIFICATION OPTION (cls_loss='vfl', default 'bce').
#           IoU-aware classification: positives are weighted by their soft TAL
#           target score (localization quality), negatives down-weighted by
#           p^gamma. Directly targets the score/IoU ranking miscalibration
#           behind high AR50_small + low AP50-95_small. vfl_alpha / vfl_gamma
#           configurable. The focusing weight uses detached predictions.
#   [NEW-5] NWD BLEND FOR SMALL OBJECTS (nwd_ratio, default 0.0 = off).
#           For small objects only, the regression loss becomes
#             (1-r)*(1-CIoU) + r*(1-NWD),
#           NWD = exp(-W2/nwd_c) with W2 the Gauss-Wasserstein distance between
#           boxes in PIXEL space (Wang et al. 2021, "A Normalized Gaussian
#           Wasserstein Distance for Tiny Object Detection"). Unlike all
#           reweighting variants (which only rescale gradients), this changes
#           the SHAPE of the loss surface where IoU is cliff-like for tiny tall
#           boxes. nwd_c should be near the dataset's typical absolute box size
#           at training resolution (default 64; mean box at 640 here is ~41x90
#           -> sqrt(w*h) ~ 61).
#           [NEW-5b] ADAPTIVE NWD BLEND. When nwd_adaptive=1, the NWD ratio is
#           modulated by object "smallness": objects right at the threshold get
#           nearly pure CIoU, while the tiniest objects get the full nwd_ratio.
#             effective_r = nwd_ratio * (px_threshold^2 - area_px) / px_threshold^2
#           This avoids the hard on/off behavior of the original flat blend.
#           [NEW-5c] NWD ANNEALING. When nwd_anneal=1, the effective NWD ratio
#           is further scaled by a linear ramp: full ratio at epoch 0, decaying
#           to nwd_anneal_min fraction by the end of training. NWD stabilizes
#           the early loss landscape; CIoU's tighter optimum dominates late.
#           Off-switch: nwd_anneal=0 (default) keeps the ratio constant.
#   [NEW-6] E2E topk guard: E2EDetectLoss's one2one head now FORCES tal_topk=1
#           (force_tal_topk); previously a tal_topk hyp silently overrode it.
#   [NEW-7] DetectObjLoss now passes stride_tensor to bbox_loss (pixel-space
#           weighting active, was silently in the grid-units fallback) and
#           applies the center loss, so it differs from the base loss ONLY by
#           the objectness term.
#   [NEW-8] Center loss decoupled from hyp.box: it is added AFTER the box gain,
#           so center_loss_weight_init means what it says (was silently
#           multiplied by hyp.box=7.5). Decay now interpolates init -> min
#           (was init -> 0 floored at min, which had a kink).
#   [NEW-9] IoU-AWARE REGRESSION WEIGHTING (IARW, Section F).
#           Per-anchor regression boost proportional to the localization deficit:
#             reg_boost = 1 + iarw_gamma * (1 - IoU).detach()
#           Applied to BOTH IoU loss and DFL loss. Predictions with high IoU
#           (already tight) get minimal boost; those with low IoU (loose boxes)
#           get amplified. Unlike area-based weighting (which assumes small=hard),
#           IARW directly *measures* which predictions need more work and is
#           self-correcting: as the box improves, the boost fades. This targets
#           the core diagnosed deficit (high AR, low AP50-95) without assumptions
#           about object size. Off-switch: iarw_gamma=0.0 (default).
#
# REPRODUCIBILITY NOTE: with all sections off (alpha=0, boost=1, dfl_small_boost=1,
# nwd_ratio=0, cls_loss='bce', clips at 999, iarw_gamma=0) this file is
# numerically identical to the previous version's anchor path. NEW-1/NEW-2 only
# alter behavior when Section A weighting is actually active.
#
# NOTE ON CUSTOM KEYS: all new params (incl. the STRING param cls_loss) must be
# whitelisted in your cfg patch exactly like the existing custom keys, or
# model.train() will reject / silently drop them. VERIFY via the config banner.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.ops import xywh2xyxy, xyxy2xywh, crop_mask
from ultralytics.utils.tal import RotatedTaskAlignedAssigner, TaskAlignedAssigner, dist2bbox, dist2rbox, make_anchors
from ultralytics.utils.torch_utils import autocast
from ultralytics.utils.metrics import OKS_SIGMA

from .metrics import bbox_iou, probiou
from .tal import bbox2dist


class VarifocalLoss(nn.Module):
    """Varifocal loss by Zhang et al."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with autocast(enabled=False):
            loss = (
                (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(), reduction="none") * weight)
                .mean(1)
                .sum()
            )
        return loss


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, gamma=1.5, alpha=0.25, reduction='sum'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, pred, label, gamma=None, alpha=None):
        gamma = gamma if gamma is not None else self.gamma
        alpha = alpha if alpha is not None else self.alpha

        loss = F.binary_cross_entropy_with_logits(pred, label, reduction="none")
        pred_prob = pred.sigmoid()
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** gamma
        loss *= modulating_factor

        if alpha > 0:
            alpha_factor = label * alpha + (1 - label) * (1 - alpha)
            loss *= alpha_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.mean(1).sum()
        return loss


class DFLoss(nn.Module):
    """Distribution Focal Loss for bounding box regression."""

    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        return (
                F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view(tl.shape) * wl
                + F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view(tl.shape) * wr
        ).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    """
    Bounding box loss with configurable ablation parameters.

    Reads parameters from model.args (passed via model.train()).

    Ablation Parameters:
        Section A (Size-aware weighting):
            - alpha_start / alpha_end / alpha_min / alpha_max
            - small_obj_px: PIXEL area-side threshold; "small" iff
              pixel area < small_obj_px^2 at training resolution. 0 disables
              the boost, dfl_small_boost, and the NWD blend.
            - small_obj_boost: area-weight multiplier for small objects
            - area_mode: 'fixed' (NEW-2, default) or 'legacy'
            - area_ref_px: reference box side for 'fixed' mode (default 64)
            - area_gamma: exponent for 'fixed' mode (0.5 = sqrt, 1.0 = inverse)
            - area_w_cap: max area weight in 'fixed' mode (default 3.0)
            - weight_renorm: 1 (NEW-1, default) renormalizes the combined
              weight sum to the score-weight sum; 0 = legacy behavior

        Section A' (targeted DFL, NEW-3 / NEW-3b):
            - dfl_small_boost: DFL-only multiplier for small objects (1.0 off)
            - dfl_iou_gated: 1 = modulate boost by (1-IoU), 0 = flat (default)

        Section A'' (NWD blend, NEW-5 / NEW-5b / NEW-5c):
            - nwd_ratio: blend ratio r for small objects (0.0 off)
            - nwd_c: NWD temperature in pixels (default 64)
            - nwd_adaptive: 1 = continuous ramp by smallness, 0 = flat (default)
            - nwd_anneal: 1 = fade NWD ratio over training, 0 = constant (default)
            - nwd_anneal_min: fraction of nwd_ratio kept at end (default 0.1)

        Section F (IARW, NEW-9):
            - iarw_gamma: IoU-aware regression boost (0.0 off, recommended 2.0-3.0)

        Section C (Adaptive clipping):
            - iou_clip_start/end, dfl_clip_start/end
            Effective per-sample cap = value / 10. Set 999 to disable.
    """

    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max) if reg_max > 1 else None
        self.reg_max = reg_max

        # Training state
        self.epoch = 0
        self.total_epochs = 70

        # Section A defaults
        self.small_obj_px = 70
        self.small_obj_boost = 1.5
        self.alpha_start = 0.9
        self.alpha_end = 0.5
        self.alpha_min = 0.3
        self.alpha_max = 0.9

        # [NEW-1] combined-weight renormalization
        self.weight_renorm = 1

        # [NEW-2] fixed-reference area weighting
        self.area_mode = 'fixed'     # 'fixed' | 'legacy'
        self.area_ref_px = 64.0
        self.area_gamma = 0.5
        self.area_w_cap = 3.0

        # [NEW-3] DFL-only small-object boost
        self.dfl_small_boost = 1.0
        # [NEW-3b] IoU-gated DFL boost
        self.dfl_iou_gated = 0

        # [NEW-5] NWD blend for small objects
        self.nwd_ratio = 0.0
        self.nwd_c = 64.0
        # [NEW-5b] Adaptive NWD blend (continuous ramp by smallness)
        self.nwd_adaptive = 0
        # [NEW-5c] NWD annealing (fade NWD over training)
        self.nwd_anneal = 0
        self.nwd_anneal_min = 0.1  # fraction of nwd_ratio kept at end of training

        # [NEW-9] IoU-Aware Regression Weighting (IARW, Section F)
        self.iarw_gamma = 0.0  # 0.0 = off; recommended 2.0-3.0

        # =====================================================================
        # ROUND 10 — regression-signal mechanisms (not reweighting)
        # =====================================================================
        # [NEW-10] alpha-IoU (power-IoU): raise the overlap term to a power > 1
        #   so gradient concentrates in the high-IoU regime (AP75/AP90). The
        #   CIoU distance+aspect penalty is preserved; only the overlap term is
        #   powered. alpha_iou=1.0 -> exact stock 1-CIoU (off).
        self.alpha_iou = 1.0

        # [NEW-11] Pixel-space residual auxiliary (Smooth-L1 / Balanced-L1).
        #   (1-IoU) gradient flattens as the box tightens; a direct L/T/R/B
        #   pixel-residual term keeps a non-vanishing gradient near the optimum,
        #   supplying the "last few pixels" signal strict-IoU AP needs.
        self.l1_aux_weight = 0.0     # 0 = off
        self.l1_aux_beta = 2.0       # smooth/balanced-L1 transition (pixels)
        self.l1_aux_small_only = 1   # 1 = small_mask only, 0 = all fg
        self.l1_balanced = 0         # 1 = Balanced-L1 (Libra), 0 = Smooth-L1
        self.l1_balanced_alpha = 0.5
        self.l1_balanced_gamma = 1.5

        # [NEW-12] DFL distribution sharpening (entropy regularizer).
        #   Penalizes the entropy of each side's softmax bin distribution, so
        #   loose/multi-modal edge distributions become sharp & unimodal ->
        #   crisper decoded edges. Shapes the prediction, not the loss weight.
        self.dfl_entropy_weight = 0.0   # 0 = off (try 0.02-0.1)
        self.dfl_entropy_small_only = 1

        # [NEW-13] Size-adaptive NWD temperature.
        #   Fixes nwd_c=64px saturating for small boxes (nwd~1 -> inert). With
        #   c = nwd_c_k * sqrt(area_px), the Gaussian scale tracks object size,
        #   so NWD stays discriminative exactly where it is meant to help.
        self.nwd_c_adaptive = 0      # 0 = fixed nwd_c; 1 = per-anchor c
        self.nwd_c_k = 0.5

        # [NEW-14] Asymmetric tightness penalty.
        #   Loose small boxes usually SPILL OVER the GT. Penalize predicted
        #   sides that extend beyond the GT box more than sides that fall short,
        #   pushing boxes to hug the object. tightness_gamma=0 -> off.
        self.tightness_gamma = 0.0
        self.tightness_small_only = 1

        # Section C: Adaptive clipping defaults
        self.iou_clip_start = 20.0
        self.iou_clip_end = 10.0
        self.dfl_clip_start = 10.0
        self.dfl_clip_end = 5.0

        # [FIX-2] Clip-rate tracking state (running average within an epoch)
        self._clip_iou_rate = 0.0
        self._clip_dfl_rate = 0.0
        self._clip_n = 0

    def set_params(self, hyp):
        """Set parameters from hyperparameters (model.args)."""
        # Section A: Size-aware weighting
        self.small_obj_px = getattr(hyp, 'small_obj_px', self.small_obj_px)
        self.small_obj_boost = getattr(hyp, 'small_obj_boost', self.small_obj_boost)
        self.alpha_start = getattr(hyp, 'alpha_start', self.alpha_start)
        self.alpha_end = getattr(hyp, 'alpha_end', self.alpha_end)
        self.alpha_min = getattr(hyp, 'alpha_min', self.alpha_min)
        self.alpha_max = getattr(hyp, 'alpha_max', self.alpha_max)
        self.total_epochs = getattr(hyp, 'epochs', self.total_epochs)

        # [NEW-1] / [NEW-2]
        self.weight_renorm = int(getattr(hyp, 'weight_renorm', self.weight_renorm))
        self.area_mode = getattr(hyp, 'area_mode', self.area_mode)
        self.area_ref_px = float(getattr(hyp, 'area_ref_px', self.area_ref_px))
        self.area_gamma = float(getattr(hyp, 'area_gamma', self.area_gamma))
        self.area_w_cap = float(getattr(hyp, 'area_w_cap', self.area_w_cap))

        # [NEW-3] / [NEW-3b]
        self.dfl_small_boost = float(getattr(hyp, 'dfl_small_boost', self.dfl_small_boost))
        self.dfl_iou_gated = int(getattr(hyp, 'dfl_iou_gated', self.dfl_iou_gated))

        # [NEW-5] / [NEW-5b] / [NEW-5c]
        self.nwd_ratio = float(getattr(hyp, 'nwd_ratio', self.nwd_ratio))
        self.nwd_c = float(getattr(hyp, 'nwd_c', self.nwd_c))
        self.nwd_adaptive = int(getattr(hyp, 'nwd_adaptive', self.nwd_adaptive))
        self.nwd_anneal = int(getattr(hyp, 'nwd_anneal', self.nwd_anneal))
        self.nwd_anneal_min = float(getattr(hyp, 'nwd_anneal_min', self.nwd_anneal_min))

        # [NEW-9] IARW
        self.iarw_gamma = float(getattr(hyp, 'iarw_gamma', self.iarw_gamma))

        # [NEW-10..14] Round-10 regression-signal mechanisms
        self.alpha_iou = float(getattr(hyp, 'alpha_iou', self.alpha_iou))
        self.l1_aux_weight = float(getattr(hyp, 'l1_aux_weight', self.l1_aux_weight))
        self.l1_aux_beta = float(getattr(hyp, 'l1_aux_beta', self.l1_aux_beta))
        self.l1_aux_small_only = int(getattr(hyp, 'l1_aux_small_only', self.l1_aux_small_only))
        self.l1_balanced = int(getattr(hyp, 'l1_balanced', self.l1_balanced))
        self.l1_balanced_alpha = float(getattr(hyp, 'l1_balanced_alpha', self.l1_balanced_alpha))
        self.l1_balanced_gamma = float(getattr(hyp, 'l1_balanced_gamma', self.l1_balanced_gamma))
        self.dfl_entropy_weight = float(getattr(hyp, 'dfl_entropy_weight', self.dfl_entropy_weight))
        self.dfl_entropy_small_only = int(getattr(hyp, 'dfl_entropy_small_only', self.dfl_entropy_small_only))
        self.nwd_c_adaptive = int(getattr(hyp, 'nwd_c_adaptive', self.nwd_c_adaptive))
        self.nwd_c_k = float(getattr(hyp, 'nwd_c_k', self.nwd_c_k))
        self.tightness_gamma = float(getattr(hyp, 'tightness_gamma', self.tightness_gamma))
        self.tightness_small_only = int(getattr(hyp, 'tightness_small_only', self.tightness_small_only))

        # Section C: Adaptive clipping
        self.iou_clip_start = getattr(hyp, 'iou_clip_start', self.iou_clip_start)
        self.iou_clip_end = getattr(hyp, 'iou_clip_end', self.iou_clip_end)
        self.dfl_clip_start = getattr(hyp, 'dfl_clip_start', self.dfl_clip_start)
        self.dfl_clip_end = getattr(hyp, 'dfl_clip_end', self.dfl_clip_end)

    def _get_dynamic_alpha(self):
        """[FIX-3] Dynamic alpha; logs alpha + clip rates once per epoch."""
        progress = self.epoch / max(self.total_epochs, 1)
        alpha = self.alpha_start * (1 - progress) + self.alpha_end * progress
        alpha = max(self.alpha_min, min(self.alpha_max, alpha))

        if not hasattr(self, '_last_logged_epoch'):
            self._last_logged_epoch = -1

        if self.epoch != self._last_logged_epoch:
            iou_r = getattr(self, "_clip_iou_rate", 0.0)
            dfl_r = getattr(self, "_clip_dfl_rate", 0.0)
            print(f"[Loss] Epoch {self.epoch}/{self.total_epochs}: "
                  f"alpha={alpha:.3f} | clip-rate iou={iou_r * 100:.2f}% dfl={dfl_r * 100:.2f}%")
            # reset running averages for the new epoch
            self._clip_iou_rate = 0.0
            self._clip_dfl_rate = 0.0
            self._clip_n = 0
            self._last_logged_epoch = self.epoch

        return alpha

    def _compute_target_areas(self, target_bboxes, fg_mask):
        """Compute target bounding box areas (grid units) with numerical stability."""
        areas = (target_bboxes[..., 2] - target_bboxes[..., 0]) * \
                (target_bboxes[..., 3] - target_bboxes[..., 1])
        return areas.clamp(min=1e-6)

    def _compute_weights(self, target_bboxes, target_scores, fg_mask, stride=None):
        """[FIX-1] Compute area and score weights in PIXEL space.

        target_bboxes arrive already divided by stride_tensor (grid units of
        each anchor's own level); areas are converted back to pixels with the
        per-anchor stride before size weighting or thresholding.

        Returns:
            score_weight: (n_fg, 1) TAL soft-score weight (stock YOLO weight)
            area_weight:  (n_fg, 1) size weight (mode-dependent), incl. boost
            small_mask:   (n_fg,) bool, area < small_obj_px^2, or None if
                          small_obj_px <= 0 / no fg samples
            stride_fg:    (n_fg, 1) per-sample stride in pixels, or None if
                          no stride tensor was supplied (pose/seg fallback)
            fg_areas_px:  (n_fg,) pixel areas per fg anchor (for adaptive NWD)
        """
        target_areas = self._compute_target_areas(target_bboxes, fg_mask)  # grid units

        score_weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)

        stride_fg = None
        if stride is not None:
            # per-anchor stride aligned with fg anchors: (n_fg,)
            stride_fg_flat = stride.view(1, -1).expand(fg_mask.shape)[fg_mask]
            fg_areas_px = target_areas[fg_mask] * stride_fg_flat.pow(2)  # grid -> pixels
            stride_fg = stride_fg_flat.unsqueeze(-1)
        else:
            # fallback (callers that pass no stride): grid units
            fg_areas_px = target_areas[fg_mask]

        # [NEW-2] size weight
        if self.area_mode == 'legacy':
            area_weight = (1.0 / fg_areas_px.clamp(min=1.0)).unsqueeze(-1)
            if area_weight.numel() > 0:
                area_weight = area_weight / (area_weight.max() + 1e-8)
        else:  # 'fixed': deterministic per-object weight, batch-independent
            area_weight = (self.area_ref_px ** 2 / fg_areas_px.clamp(min=1.0)) \
                .pow(self.area_gamma).clamp(max=self.area_w_cap).unsqueeze(-1)

        # Small-object mask (shared by boost, dfl_small_boost, NWD blend)
        small_mask = None
        if fg_areas_px.numel() > 0 and self.small_obj_px > 0:
            small_mask = fg_areas_px < float(self.small_obj_px) ** 2

        # Small object boost on the area weight
        if small_mask is not None and self.small_obj_boost != 1.0 and small_mask.any():
            area_weight = area_weight.clone()
            area_weight[small_mask] *= self.small_obj_boost

        return score_weight, area_weight, small_mask, stride_fg, fg_areas_px

    def _get_gradient_clip_values(self):
        """Get adaptive clipping values based on training progress."""
        progress = self.epoch / max(self.total_epochs, 1)
        max_iou = self.iou_clip_end + (self.iou_clip_start - self.iou_clip_end) * (1 - progress)
        max_dfl = self.dfl_clip_end + (self.dfl_clip_start - self.dfl_clip_end) * (1 - progress)
        return max_iou, max_dfl

    def _get_effective_nwd_ratio(self):
        """[NEW-5c] NWD ratio with optional epoch annealing.

        Returns the base nwd_ratio, optionally scaled down over training so
        that NWD dominates early (smooth landscape) and CIoU dominates late
        (tight optimum). Off-switch: nwd_anneal=0 returns nwd_ratio unchanged.
        """
        r = self.nwd_ratio
        if self.nwd_anneal and self.total_epochs > 0:
            progress = self.epoch / max(self.total_epochs, 1)
            # Linear decay: 1.0 at epoch 0 -> nwd_anneal_min at final epoch
            anneal_factor = 1.0 - (1.0 - self.nwd_anneal_min) * progress
            r = r * anneal_factor
        return r

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride=None):
        """Compute IoU(+NWD) and DFL losses with IARW, adaptive NWD, IoU-gated
        DFL boost, per-sample clipping, and clip-rate logging."""

        alpha = self._get_dynamic_alpha()
        score_weight, area_weight, small_mask, stride_fg, fg_areas_px = \
            self._compute_weights(target_bboxes, target_scores, fg_mask, stride)

        # Combined weight
        weight = alpha * area_weight + (1 - alpha) * score_weight

        # [NEW-1] renormalize: redistribute across samples, preserve magnitude.
        # Identity when alpha == 0 (weight == score_weight), so anchors are
        # unaffected; makes Section-A components composable (r7_stack lesson).
        if self.weight_renorm and weight.numel() > 0:
            weight = weight * (score_weight.sum() / (weight.sum() + 1e-8))

        # IoU loss per sample
        pred_fg = pred_bboxes[fg_mask]
        targ_fg = target_bboxes[fg_mask]
        iou = bbox_iou(pred_fg, targ_fg, xywh=False, CIoU=True)
        loss_reg = 1.0 - iou  # (n_fg, 1)

        # [NEW-10] alpha-IoU (power-IoU): power ONLY the overlap term, keep the
        # CIoU distance+aspect penalty intact. Concentrates gradient at high
        # IoU -> directly targets strict-threshold (AP75/AP90) localization.
        # Off-switch: alpha_iou == 1.0 -> loss_reg unchanged.
        if self.alpha_iou != 1.0:
            raw_iou = bbox_iou(pred_fg, targ_fg, xywh=False).clamp(1e-7, 1.0)  # plain IoU
            ciou_penalty = (raw_iou - iou).clamp(min=0.0)  # distance+aspect part of CIoU
            loss_reg = (1.0 - raw_iou.pow(self.alpha_iou)) + ciou_penalty

        # [NEW-5] NWD blend for small objects (pixel space; needs stride_fg)
        # [NEW-5b] Adaptive: continuous ramp by smallness instead of flat ratio
        # [NEW-5c] Annealing: fade NWD over training epochs
        effective_nwd_ratio = self._get_effective_nwd_ratio()
        if (effective_nwd_ratio > 0.0 and small_mask is not None and
                stride_fg is not None and small_mask.any()):
            p_px = pred_fg * stride_fg
            t_px = targ_fg * stride_fg
            pc = (p_px[:, :2] + p_px[:, 2:]) / 2
            tc = (t_px[:, :2] + t_px[:, 2:]) / 2
            pwh = (p_px[:, 2:] - p_px[:, :2]).clamp(min=0)
            twh = (t_px[:, 2:] - t_px[:, :2]).clamp(min=0)
            # Gauss-Wasserstein^2 between N(c, diag((w/2)^2,(h/2)^2))
            w2 = (pc - tc).pow(2).sum(-1) + ((pwh - twh) / 2).pow(2).sum(-1)
            dist = w2.clamp(min=1e-7).sqrt()  # (n_fg,)
            # [NEW-13] size-adaptive temperature: c tracks object size so NWD
            # does not saturate (nwd~1) for the small boxes it targets. With a
            # fixed nwd_c=64px, sqrt(w2)/64 ~ 0 for tiny boxes -> inert.
            if self.nwd_c_adaptive:
                c = (self.nwd_c_k * fg_areas_px.clamp(min=1.0).sqrt()).clamp(min=4.0)  # (n_fg,)
            else:
                c = self.nwd_c
            nwd = torch.exp(-dist / c).unsqueeze(-1)

            loss_reg = loss_reg.clone()

            if self.nwd_adaptive and self.small_obj_px > 0:
                # [NEW-5b] Per-anchor NWD ratio proportional to "smallness":
                # objects at the threshold edge get ~0 NWD (pure CIoU);
                # the tiniest objects get the full effective_nwd_ratio.
                threshold_sq = float(self.small_obj_px) ** 2
                smallness = ((threshold_sq - fg_areas_px) / threshold_sq).clamp(0, 1)
                per_anchor_r = effective_nwd_ratio * smallness[small_mask].unsqueeze(-1)
                loss_reg[small_mask] = ((1.0 - per_anchor_r) * loss_reg[small_mask]
                                        + per_anchor_r * (1.0 - nwd[small_mask]))
            else:
                # Original flat blend for all small objects
                r = effective_nwd_ratio
                loss_reg[small_mask] = ((1.0 - r) * loss_reg[small_mask]
                                        + r * (1.0 - nwd[small_mask]))

        # [NEW-9] IARW: IoU-Aware Regression Weighting (Section F)
        # Boost regression loss proportionally to the localization deficit.
        # Self-correcting: as the box tightens (IoU rises), the boost fades.
        # Off-switch: iarw_gamma=0.0 -> iarw_boost = 1.0 everywhere (identity).
        if self.iarw_gamma > 0.0:
            with torch.no_grad():
                iou_deficit = (1.0 - iou).clamp(min=0.0)          # (n_fg, 1)
                iarw_boost = 1.0 + self.iarw_gamma * iou_deficit   # >= 1.0
            loss_reg = loss_reg * iarw_boost

        per_sample_iou_loss = loss_reg * weight

        # Get adaptive clip values (effective per-sample cap = value / 10)
        max_iou_clip, max_dfl_clip = self._get_gradient_clip_values()
        iou_cap = max_iou_clip / 10.0

        # [FIX-2] clip-rate tracking: fraction of fg samples hitting the cap
        with torch.no_grad():
            if per_sample_iou_loss.numel() > 0:
                clipped = (per_sample_iou_loss > iou_cap).float().mean().item()
                n = self._clip_n
                self._clip_iou_rate = (self._clip_iou_rate * n + clipped) / (n + 1)

        # Clip PER-SAMPLE
        per_sample_iou_loss = per_sample_iou_loss.clamp(max=iou_cap)

        # Aggregate
        loss_iou = per_sample_iou_loss.sum() / target_scores_sum

        # [NEW-11] Pixel-space residual auxiliary (Smooth-L1 / Balanced-L1).
        # Added AFTER the per-sample clip so it is not swallowed by the IoU cap,
        # and with its own small weight. Non-vanishing gradient near the optimum
        # is exactly the "last few pixels" signal (1-IoU) fails to provide.
        if self.l1_aux_weight > 0.0:
            pf_px = pred_fg * stride_fg if stride_fg is not None else pred_fg
            tf_px = targ_fg * stride_fg if stride_fg is not None else targ_fg
            beta = max(self.l1_aux_beta, 1e-6)
            x = (pf_px - tf_px).abs() / beta  # (n_fg, 4) in beta-pixel units
            if self.l1_balanced:
                a, g = self.l1_balanced_alpha, self.l1_balanced_gamma
                b_ = math.exp(g / a) - 1.0
                l1 = torch.where(x < 1.0,
                                 (a / b_) * (b_ * x + 1.0) * torch.log(b_ * x + 1.0) - a * x,
                                 g * x + (g / b_ - a))
            else:  # Smooth-L1
                l1 = torch.where(x < 1.0, 0.5 * x * x, x - 0.5)
            l1 = l1.mean(dim=1, keepdim=True)  # (n_fg, 1)
            if self.l1_aux_small_only and small_mask is not None:
                l1 = l1 * small_mask.unsqueeze(-1)
            loss_iou = loss_iou + self.l1_aux_weight * (l1 * weight).sum() / target_scores_sum

        # [NEW-14] Asymmetric tightness penalty: charge the sides where the
        # prediction SPILLS OVER the GT (loose boxes overshoot), normalized by
        # object diagonal so it is scale-free. Pushes boxes to hug the object.
        if self.tightness_gamma > 0.0:
            pf_px = pred_fg * stride_fg if stride_fg is not None else pred_fg
            tf_px = targ_fg * stride_fg if stride_fg is not None else targ_fg
            over = torch.stack([
                (tf_px[:, 0] - pf_px[:, 0]).clamp(min=0.0),  # pred beyond gt-left
                (tf_px[:, 1] - pf_px[:, 1]).clamp(min=0.0),  # pred beyond gt-top
                (pf_px[:, 2] - tf_px[:, 2]).clamp(min=0.0),  # pred beyond gt-right
                (pf_px[:, 3] - tf_px[:, 3]).clamp(min=0.0),  # pred beyond gt-bottom
            ], dim=-1)
            diag = (tf_px[:, 2:] - tf_px[:, :2]).clamp(min=1.0).norm(dim=-1, keepdim=True)
            tight = over.sum(-1, keepdim=True) / diag  # (n_fg, 1)
            if self.tightness_small_only and small_mask is not None:
                tight = tight * small_mask.unsqueeze(-1)
            loss_iou = loss_iou + self.tightness_gamma * (tight * weight).sum() / target_scores_sum

        # DFL loss per sample
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            per_sample_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask]
            ) * weight

            # [NEW-9] IARW on DFL too: same boost, consistent signal
            if self.iarw_gamma > 0.0:
                per_sample_dfl = per_sample_dfl * iarw_boost

            # [NEW-3] DFL-only boost for small objects, applied BEFORE the cap
            # (so a tight Section-C cap can absorb it -- watch the clip-rate log)
            # [NEW-3b] IoU-gated: modulate by (1-IoU) so only loose boxes get
            # the full boost; well-localized small objects are left alone.
            if (self.dfl_small_boost != 1.0 and small_mask is not None
                    and small_mask.any()):
                per_sample_dfl = per_sample_dfl.clone()
                if self.dfl_iou_gated:
                    with torch.no_grad():
                        iou_gate = (1.0 - iou).clamp(0.0, 1.0)  # (n_fg, 1)
                    effective_boost = 1.0 + (self.dfl_small_boost - 1.0) * iou_gate[small_mask]
                    per_sample_dfl[small_mask] *= effective_boost
                else:
                    per_sample_dfl[small_mask] *= self.dfl_small_boost

            dfl_cap = max_dfl_clip / 10.0

            # [FIX-2] clip-rate tracking
            with torch.no_grad():
                if per_sample_dfl.numel() > 0:
                    clipped = (per_sample_dfl > dfl_cap).float().mean().item()
                    n = self._clip_n
                    self._clip_dfl_rate = (self._clip_dfl_rate * n + clipped) / (n + 1)
                self._clip_n += 1

            # Clip PER-SAMPLE
            per_sample_dfl = per_sample_dfl.clamp(max=dfl_cap)

            # Aggregate
            loss_dfl = per_sample_dfl.sum() / target_scores_sum

            # [NEW-12] DFL distribution sharpening: penalize per-side softmax
            # entropy so edge distributions become sharp & unimodal (crisper
            # decoded edges). Targets the distribution shape, not the weight.
            if self.dfl_entropy_weight > 0.0:
                p = pred_dist[fg_mask].view(-1, 4, self.reg_max).softmax(-1)
                ent = -(p * p.clamp_min(1e-9).log()).sum(-1).mean(-1, keepdim=True)  # (n_fg,1)
                if self.dfl_entropy_small_only and small_mask is not None:
                    ent = ent * small_mask.unsqueeze(-1)
                loss_dfl = loss_dfl + self.dfl_entropy_weight * (ent * weight).sum() / target_scores_sum
        else:
            self._clip_n += 1
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


class RotatedBboxLoss(BboxLoss):
    """Criterion class for computing rotated bounding box losses."""

    def __init__(self, reg_max):
        super().__init__(reg_max)

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride=None):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = probiou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(
                anchor_points,
                xywh2xyxy(target_bboxes[..., :4]),
                self.dfl_loss.reg_max - 1
            )
            loss_dfl = self.dfl_loss(
                pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max),
                target_ltrb[fg_mask]
            ) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0, device=pred_dist.device)

        return loss_iou, loss_dfl


class KeypointLoss(nn.Module):
    """Criterion class for computing keypoint losses."""

    def __init__(self, sigmas):
        super().__init__()
        self.sigmas = sigmas

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        d = (pred_kpts[..., 0] - gt_kpts[..., 0]).pow(2) + (pred_kpts[..., 1] - gt_kpts[..., 1]).pow(2)
        kpt_loss_factor = kpt_mask.shape[1] / (torch.sum(kpt_mask != 0, dim=1) + 1e-9)
        e = d / ((2 * self.sigmas).pow(2) * (area + 1e-9) * 2)
        return (kpt_loss_factor.view(-1, 1) * ((1 - torch.exp(-e)) * kpt_mask)).mean()


class v8DetectionLoss:
    """
    YOLOv8 Detection Loss with configurable ablation parameters.

    All custom parameters are read from model.args (hyperparameters).
    Pass them via model.train(..., param_name=value, ...).

    Ablation Parameters:
        Section A (Size-aware weighting):
            - alpha_start, alpha_end, alpha_min, alpha_max
            - small_obj_px, small_obj_boost
            - area_mode, area_ref_px, area_gamma, area_w_cap    [NEW-2]
            - weight_renorm                                      [NEW-1]

        Section A' (DFL small boost):
            - dfl_small_boost                                    [NEW-3]
            - dfl_iou_gated                                      [NEW-3b]

        Section A'' (NWD blend):
            - nwd_ratio, nwd_c                                   [NEW-5]
            - nwd_adaptive                                       [NEW-5b]
            - nwd_anneal, nwd_anneal_min                         [NEW-5c]

        Section B (Center loss):
            - center_loss_weight_init / center_loss_weight_min
            - center_loss_decay_epochs
            NOTE: shares small_obj_px with Section A. If Section A weighting
            is off (alpha=0, boost=1.0), small_obj_px must still be > 0 for
            the center loss / dfl_small_boost / NWD blend to fire.
            [NEW-8] the center loss is added AFTER the box gain, so its weight
            is no longer silently multiplied by hyp.box.

        Section C (Adaptive clipping):
            - iou_clip_start/end, dfl_clip_start/end

        Section D (TAL assignment):
            - tal_topk, tal_alpha, tal_beta

        Section E (Classification, NEW-4):
            - cls_loss: 'bce' (default) or 'vfl'
            - vfl_alpha (default 0.75), vfl_gamma (default 2.0)

        Section F (IARW, NEW-9):
            - iarw_gamma (0.0 off, recommended 2.0-3.0)
    """

    def __init__(self, model, tal_topk=10, force_tal_topk=False):
        """Initialize v8DetectionLoss with parameters from model.args.

        Args:
            model: the detection model (unwrapped).
            tal_topk: default TAL topk when not overridden by hyp.
            force_tal_topk: [NEW-6] if True, USE the constructor tal_topk and
                ignore any tal_topk hyperparameter. Required by E2EDetectLoss's
                one2one head, whose topk=1 was previously silently overridden
                by Section D sweeps.
        """

        device = next(model.parameters()).device
        h = model.args  # Hyperparameters from model.train()
        m = model.model[-1]  # Detect() module
        self._model = model  # Store model reference for epoch sync

        # Model properties
        self.device = device
        self.hyp = h
        self.stride = m.stride
        self.nc = m.nc
        self.no = m.nc + m.reg_max * 4
        self.reg_max = m.reg_max
        self.use_dfl = m.reg_max > 1

        # =====================================================================
        # READ ABLATION PARAMETERS FROM model.args
        # =====================================================================

        # Training state
        self.epoch = 0
        self.total_epochs = getattr(h, 'epochs', 70)

        # Section A: Size-aware weighting
        self.small_obj_px = getattr(h, 'small_obj_px', 70)
        self.small_obj_boost = getattr(h, 'small_obj_boost', 1.5)
        self.alpha_start = getattr(h, 'alpha_start', 0.9)
        self.alpha_end = getattr(h, 'alpha_end', 0.5)
        self.alpha_min = getattr(h, 'alpha_min', 0.3)
        self.alpha_max = getattr(h, 'alpha_max', 0.9)
        self.weight_renorm = int(getattr(h, 'weight_renorm', 1))
        self.area_mode = getattr(h, 'area_mode', 'fixed')
        self.area_ref_px = float(getattr(h, 'area_ref_px', 64.0))
        self.area_gamma = float(getattr(h, 'area_gamma', 0.5))
        self.area_w_cap = float(getattr(h, 'area_w_cap', 3.0))
        self.dfl_small_boost = float(getattr(h, 'dfl_small_boost', 1.0))
        self.dfl_iou_gated = int(getattr(h, 'dfl_iou_gated', 0))
        self.nwd_ratio = float(getattr(h, 'nwd_ratio', 0.0))
        self.nwd_c = float(getattr(h, 'nwd_c', 64.0))
        self.nwd_adaptive = int(getattr(h, 'nwd_adaptive', 0))
        self.nwd_anneal = int(getattr(h, 'nwd_anneal', 0))
        self.nwd_anneal_min = float(getattr(h, 'nwd_anneal_min', 0.1))
        self.iarw_gamma = float(getattr(h, 'iarw_gamma', 0.0))

        # Round-10 regression-signal mechanisms (banner only; used in BboxLoss)
        self.alpha_iou = float(getattr(h, 'alpha_iou', 1.0))
        self.l1_aux_weight = float(getattr(h, 'l1_aux_weight', 0.0))
        self.l1_balanced = int(getattr(h, 'l1_balanced', 0))
        self.dfl_entropy_weight = float(getattr(h, 'dfl_entropy_weight', 0.0))
        self.nwd_c_adaptive = int(getattr(h, 'nwd_c_adaptive', 0))
        self.tightness_gamma = float(getattr(h, 'tightness_gamma', 0.0))

        # Section B: Center loss
        self.center_loss_weight_init = getattr(h, 'center_loss_weight_init', 0.0)
        self.center_loss_weight_min = getattr(h, 'center_loss_weight_min', 0.01)
        self.center_loss_decay_epochs = getattr(h, 'center_loss_decay_epochs', 35)

        # Section C: Adaptive clipping (read here for _print_config)
        self.iou_clip_start = getattr(h, 'iou_clip_start', 20.0)
        self.iou_clip_end = getattr(h, 'iou_clip_end', 10.0)
        self.dfl_clip_start = getattr(h, 'dfl_clip_start', 10.0)
        self.dfl_clip_end = getattr(h, 'dfl_clip_end', 5.0)

        # Section D: TAL parameters  [NEW-6]
        self.tal_topk = tal_topk if force_tal_topk else getattr(h, 'tal_topk', tal_topk)
        self.tal_alpha = getattr(h, 'tal_alpha', 0.5)
        self.tal_beta = getattr(h, 'tal_beta', 6.0)

        # Section E: Classification  [NEW-4]
        self.cls_loss = str(getattr(h, 'cls_loss', 'bce')).lower()
        self.vfl_alpha = float(getattr(h, 'vfl_alpha', 0.75))
        self.vfl_gamma = float(getattr(h, 'vfl_gamma', 2.0))

        # =====================================================================
        # LOSS FUNCTIONS
        # =====================================================================

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.bbox_loss = BboxLoss(m.reg_max).to(device)

        # Pass parameters to BboxLoss
        self.bbox_loss.set_params(h)

        # Task Aligned Assigner with configurable parameters
        self.assigner = TaskAlignedAssigner(
            topk=self.tal_topk,
            num_classes=self.nc,
            alpha=self.tal_alpha,
            beta=self.tal_beta
        )

        # Projection for DFL
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

        # Print configuration (only once)
        self._print_config()

    def _print_config(self):
        """Print current configuration for verification."""
        if not hasattr(self, '_config_printed'):
            print("\n" + "=" * 60)
            print("v8DetectionLoss Configuration")
            print("=" * 60)
            print(f"  [A] alpha_start:     {self.alpha_start}")
            print(f"  [A] alpha_end:       {self.alpha_end}")
            print(f"  [A] alpha_min/max:   {self.alpha_min} / {self.alpha_max}")
            print(f"  [A] small_obj_px:    {self.small_obj_px}  (pixel area threshold: area < px^2)")
            print(f"  [A] small_obj_boost: {self.small_obj_boost}")
            print(f"  [A] weight_renorm:   {self.weight_renorm}  (NEW-1)")
            print(f"  [A] area_mode:       {self.area_mode}  (NEW-2)"
                  f"{'' if self.area_mode == 'legacy' else f'  ref={self.area_ref_px}px gamma={self.area_gamma} cap={self.area_w_cap}'}")
            print(f"  [A'] dfl_small_boost: {self.dfl_small_boost}  (NEW-3)"
                  f"{'  IoU-gated (NEW-3b)' if self.dfl_iou_gated else ''}")
            print(f"  [A''] nwd_ratio/c:   {self.nwd_ratio} / {self.nwd_c}px  (NEW-5)"
                  f"{'  adaptive (NEW-5b)' if self.nwd_adaptive else ''}"
                  f"{'  anneal->%.2f (NEW-5c)' % self.nwd_anneal_min if self.nwd_anneal else ''}")
            print(f"  [F] iarw_gamma:      {self.iarw_gamma}  (NEW-9, 0=off)")
            print(f"  [R10] alpha_iou:     {self.alpha_iou}  (NEW-10, 1.0=off)")
            print(f"  [R10] l1_aux_weight: {self.l1_aux_weight}  (NEW-11, 0=off"
                  f"{', Balanced-L1' if self.l1_balanced else ', Smooth-L1'})")
            print(f"  [R10] dfl_entropy:   {self.dfl_entropy_weight}  (NEW-12, 0=off)")
            print(f"  [R10] nwd_c_adaptive:{self.nwd_c_adaptive}  (NEW-13)")
            print(f"  [R10] tightness_gamma:{self.tightness_gamma}  (NEW-14, 0=off)")
            print(f"  [B] center_loss_init:  {self.center_loss_weight_init}  (applied AFTER box gain, NEW-8)")
            print(f"  [B] center_loss_min:   {self.center_loss_weight_min}")
            print(f"  [B] center_loss_decay: {self.center_loss_decay_epochs} epochs")
            print(f"  [C] iou_clip:        {self.iou_clip_start} -> {self.iou_clip_end}  (eff. /10)")
            print(f"  [C] dfl_clip:        {self.dfl_clip_start} -> {self.dfl_clip_end}  (eff. /10)")
            print(f"  [D] tal_topk:        {self.tal_topk}")
            print(f"  [D] tal_alpha:       {self.tal_alpha}")
            print(f"  [D] tal_beta:        {self.tal_beta}")
            print(f"  [E] cls_loss:        {self.cls_loss}"
                  f"{'' if self.cls_loss == 'bce' else f'  (vfl_alpha={self.vfl_alpha} vfl_gamma={self.vfl_gamma})'}  (NEW-4)")
            print(f"  epochs:              {self.total_epochs}")
            print("=" * 60 + "\n")
            self._config_printed = True

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocess target counts and matches with input batch size."""
        nl, ne = targets.shape
        if nl == 0:
            return torch.zeros(batch_size, 0, ne - 1, device=self.device)

        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)

        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                out[j, :n] = targets[matches, 1:]

        out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out

    def bbox_decode(self, anchor_points, pred_dist):
        """Decode predicted bounding box coordinates."""
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )
        return dist2bbox(pred_dist, anchor_points, xywh=False)

    def _compute_cls_loss(self, pred_scores, target_scores, target_scores_sum, dtype):
        """[NEW-4] Classification loss: stock BCE or Varifocal (IoU-aware).

        VFL: positives (target_score > 0) are weighted by their soft TAL score
        (which encodes localization quality), so the classifier is trained to
        rank by IoU; negatives are down-weighted by sigmoid(p)^gamma * alpha.
        The focusing weight uses DETACHED predictions (no gradient through the
        weight), the standard stable formulation.
        """
        ts = target_scores.to(dtype)
        if self.cls_loss == 'vfl':
            label = (ts > 0).to(dtype)
            with torch.no_grad():
                pred_sig = pred_scores.sigmoid()
                w = self.vfl_alpha * pred_sig.pow(self.vfl_gamma) * (1 - label) + ts * label
            loss = (F.binary_cross_entropy_with_logits(pred_scores, ts, reduction="none") * w)
            return loss.sum() / target_scores_sum
        # default: BCE (stock)
        return self.bce(pred_scores, ts).sum() / target_scores_sum

    def _compute_center_loss(self, pred_bboxes, target_bboxes, fg_mask, stride_tensor):
        """[FIX-4] Auxiliary scale-free center loss for small objects (Section B).

        All quantities converted to PIXEL space with each anchor's own stride.
        Error is |pred_center - gt_center|_px / sqrt(target_area_px).
        [NEW-8] decay now interpolates init -> min over decay_epochs (no kink),
        and the returned value is added AFTER the box gain by the caller.
        """

        # Skip if center loss is disabled
        if self.center_loss_weight_init <= 0:
            return torch.tensor(0.0, device=self.device)

        if not fg_mask.any():
            return torch.tensor(0.0, device=self.device)

        b_idx, a_idx = torch.nonzero(fg_mask, as_tuple=True)
        if b_idx.numel() == 0:
            return torch.tensor(0.0, device=self.device)

        pred_fg = pred_bboxes[b_idx, a_idx]      # grid units (own level)
        target_fg = target_bboxes[b_idx, a_idx]  # grid units (own level)
        stride_fg = stride_tensor.view(-1)[a_idx].unsqueeze(-1)  # (n_fg, 1)

        # Centers and areas in PIXEL space
        pred_centers = (pred_fg[:, :2] + pred_fg[:, 2:]) / 2 * stride_fg
        target_centers = (target_fg[:, :2] + target_fg[:, 2:]) / 2 * stride_fg
        target_areas_px = ((target_fg[:, 2] - target_fg[:, 0]) *
                           (target_fg[:, 3] - target_fg[:, 1])) * stride_fg.squeeze(-1).pow(2)

        # Small object mask — pixel threshold shared with Section A
        small_obj_mask = target_areas_px < float(self.small_obj_px) ** 2

        if not small_obj_mask.any():
            return torch.tensor(0.0, device=self.device)

        # Scale-free center error: pixel offset / object size
        box_scale = target_areas_px[small_obj_mask].clamp(min=1.0).sqrt().unsqueeze(-1)
        center_err = (pred_centers[small_obj_mask] - target_centers[small_obj_mask]).abs() / box_scale
        center_loss = center_err.mean()

        # [NEW-8] Progressive weight decay: interpolate init -> min
        progress = min(self.epoch / max(self.center_loss_decay_epochs, 1), 1.0)
        weight = self.center_loss_weight_init * (1 - progress) + self.center_loss_weight_min * progress

        return center_loss * weight

    def _sync_bbox_loss_state(self):
        """Synchronize epoch information with bbox_loss module."""
        self.bbox_loss.epoch = self.epoch
        self.bbox_loss.total_epochs = self.total_epochs

    def __call__(self, preds, batch):
        """Calculate the sum of detection losses (box, cls, dfl)."""

        # Try to get epoch from model
        try:
            if hasattr(self._model, 'current_epoch'):
                self.epoch = self._model.current_epoch
        except:
            pass

        self._sync_bbox_loss_state()
        loss = torch.zeros(3, device=self.device)

        # Extract features
        feats = preds[1] if isinstance(preds, tuple) else preds
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        # Prepare targets
        targets = torch.cat(
            (batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1
        )
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Decode predicted boxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        # Task Aligned Assignment
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        # [NEW-4] Classification loss (BCE or VFL)
        loss[1] = self._compute_cls_loss(pred_scores, target_scores, target_scores_sum, dtype)

        # Bounding box losses
        center_loss = torch.tensor(0.0, device=self.device)
        if fg_mask.sum():
            target_bboxes /= stride_tensor

            # Sync training state
            self._sync_bbox_loss_state()

            # IoU and DFL losses
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor
            )

            # Auxiliary center loss for small objects (Section B)
            center_loss = self._compute_center_loss(
                pred_bboxes, target_bboxes, fg_mask, stride_tensor
            )

        # Apply loss gains
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        # [NEW-8] center loss added AFTER the box gain so its weight is literal
        loss[0] = loss[0] + center_loss

        return loss.sum() * batch_size, loss.detach()


class v8ClassificationLoss:
    """Criterion class for computing classification training losses."""

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        loss = F.cross_entropy(preds, batch["cls"], reduction="mean")
        return loss, loss.detach()


class v8OBBLoss(v8DetectionLoss):
    """Calculates losses for oriented bounding box (OBB) detection."""

    def __init__(self, model):
        super().__init__(model)
        self.assigner = RotatedTaskAlignedAssigner(
            topk=self.tal_topk,
            num_classes=self.nc,
            alpha=self.tal_alpha,
            beta=self.tal_beta
        )
        self.bbox_loss = RotatedBboxLoss(self.reg_max).to(self.device)
        self.focal_loss = FocalLoss(gamma=1.5, alpha=0.25)

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            return torch.zeros(batch_size, 0, 6, device=self.device)

        i = targets[:, 0]
        _, counts = i.unique(return_counts=True)
        counts = counts.to(dtype=torch.int32)
        out = torch.zeros(batch_size, counts.max(), 6, device=self.device)

        for j in range(batch_size):
            matches = i == j
            n = matches.sum()
            if n:
                bboxes = targets[matches, 2:]
                bboxes[..., :4].mul_(scale_tensor)
                out[j, :n] = torch.cat([targets[matches, 1:2], bboxes], dim=-1)

        return out

    def bbox_decode(self, anchor_points, pred_dist, pred_angle):
        if self.use_dfl:
            b, a, c = pred_dist.shape
            pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3).matmul(
                self.proj.type(pred_dist.dtype)
            )
        return torch.cat((dist2rbox(pred_dist, pred_angle, anchor_points), pred_angle), dim=-1)

    def __call__(self, preds, batch):
        loss = torch.zeros(3, device=self.device)

        feats, pred_angle = preds if isinstance(preds[0], list) else preds[1]
        batch_size = pred_angle.shape[0]

        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_angle = pred_angle.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat(
                (batch_idx, batch["cls"].view(-1, 1), batch["bboxes"].view(-1, 5)), 1
            )
            rw = targets[:, 4] * imgsz[0].item()
            rh = targets[:, 5] * imgsz[1].item()
            targets = targets[(rw >= 2) & (rh >= 2)]
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 5), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR ❌ OBB dataset incorrectly formatted.") from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri, pred_angle)

        bboxes_for_assigner = pred_bboxes.clone().detach()
        bboxes_for_assigner[..., :4] *= stride_tensor

        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            bboxes_for_assigner.type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[1] = self.focal_loss(pred_scores, target_scores.to(dtype)) / target_scores_sum

        if fg_mask.sum():
            target_bboxes[..., :4] /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor
            )
        else:
            loss[0] += (pred_angle * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()


class E2EDetectLoss:
    """End-to-end detection loss.

    [NEW-6] one2one FORCES tal_topk=1 (force_tal_topk=True): a tal_topk
    hyperparameter no longer silently overrides it. The one2many head still
    respects Section-D sweeps.
    """

    def __init__(self, model):
        self.one2many = v8DetectionLoss(model, tal_topk=10)
        self.one2one = v8DetectionLoss(model, tal_topk=1, force_tal_topk=True)

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, tuple) else preds

        one2many = preds["one2many"]
        loss_one2many = self.one2many(one2many, batch)

        one2one = preds["one2one"]
        loss_one2one = self.one2one(one2one, batch)

        return (
            loss_one2many[0] + loss_one2one[0],
            loss_one2many[1] + loss_one2one[1]
        )


class v8PoseLoss(v8DetectionLoss):
    """Criterion class for computing pose losses."""

    def __init__(self, model):
        super().__init__(model)
        self.kpt_shape = model.model[-1].kpt_shape
        self.bce_pose = nn.BCEWithLogitsLoss()
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        sigmas = torch.from_numpy(OKS_SIGMA).to(self.device) if is_pose else torch.ones(nkpt, device=self.device) / nkpt
        self.keypoint_loss = KeypointLoss(sigmas=sigmas)

    def __call__(self, preds, batch):
        loss = torch.zeros(5, device=self.device)
        feats, pred_kpts = preds if isinstance(preds[0], list) else preds[1]
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_kpts = pred_kpts.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        batch_size = pred_scores.shape[0]
        batch_idx = batch["batch_idx"].view(-1, 1)
        targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pred_kpts = self.kpts_decode(anchor_points, pred_kpts.view(batch_size, -1, *self.kpt_shape))

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[3] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            target_bboxes /= stride_tensor
            # pass stride_tensor so pixel-space weighting is active (was the
            # grid-units fallback before)
            loss[0], loss[4] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor
            )
            keypoints = batch["keypoints"].to(self.device).float().clone()
            keypoints[..., 0] *= imgsz[1]
            keypoints[..., 1] *= imgsz[0]

            loss[1], loss[2] = self.calculate_keypoints_loss(
                fg_mask, target_gt_idx, keypoints, batch_idx, stride_tensor, target_bboxes, pred_kpts
            )

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.pose
        loss[2] *= self.hyp.kobj
        loss[3] *= self.hyp.cls
        loss[4] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def kpts_decode(anchor_points, pred_kpts):
        y = pred_kpts.clone()
        y[..., :2] *= 2.0
        y[..., 0] += anchor_points[:, [0]] - 0.5
        y[..., 1] += anchor_points[:, [1]] - 0.5
        return y

    def calculate_keypoints_loss(self, masks, target_gt_idx, keypoints, batch_idx,
                                 stride_tensor, target_bboxes, pred_kpts):
        batch_idx = batch_idx.flatten()
        batch_size = len(masks)

        max_kpts = torch.unique(batch_idx, return_counts=True)[1].max()

        batched_keypoints = torch.zeros(
            (batch_size, max_kpts, keypoints.shape[1], keypoints.shape[2]), device=keypoints.device
        )

        for i in range(batch_size):
            keypoints_i = keypoints[batch_idx == i]
            batched_keypoints[i, : keypoints_i.shape[0]] = keypoints_i

        target_gt_idx_expanded = target_gt_idx.unsqueeze(-1).unsqueeze(-1)

        selected_keypoints = batched_keypoints.gather(
            1, target_gt_idx_expanded.expand(-1, -1, keypoints.shape[1], keypoints.shape[2])
        )

        selected_keypoints /= stride_tensor.view(1, -1, 1, 1)

        kpts_loss = 0
        kpts_obj_loss = 0

        if masks.any():
            gt_kpt = selected_keypoints[masks]
            area = xyxy2xywh(target_bboxes[masks])[:, 2:].prod(1, keepdim=True)
            pred_kpt = pred_kpts[masks]
            kpt_mask = gt_kpt[..., 2] != 0 if gt_kpt.shape[-1] == 3 else torch.full_like(gt_kpt[..., 0], True)
            kpts_loss = self.keypoint_loss(pred_kpt, gt_kpt, kpt_mask, area)

            if pred_kpt.shape[-1] == 3:
                kpts_obj_loss = self.bce_pose(pred_kpt[..., 2], kpt_mask.float())

        return kpts_loss, kpts_obj_loss


class v8SegmentationLoss(v8DetectionLoss):
    """Criterion class for computing segmentation losses."""

    def __init__(self, model):
        super().__init__(model)
        self.overlap = model.args.overlap_mask

    def __call__(self, preds, batch):
        loss = torch.zeros(4, device=self.device)
        feats, pred_masks, proto = preds if len(preds) == 3 else preds[1]
        batch_size, _, mask_h, mask_w = proto.shape
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_masks = pred_masks.permute(0, 2, 1).contiguous()

        dtype = pred_scores.dtype
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        try:
            batch_idx = batch["batch_idx"].view(-1, 1)
            targets = torch.cat((batch_idx, batch["cls"].view(-1, 1), batch["bboxes"]), 1)
            targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
            gt_labels, gt_bboxes = targets.split((1, 4), 2)
            mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        except RuntimeError as e:
            raise TypeError("ERROR ❌ segment dataset incorrectly formatted.") from e

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)

        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)

        loss[2] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum

        if fg_mask.sum():
            # pass stride_tensor so pixel-space weighting is active (was the
            # grid-units fallback before)
            loss[0], loss[3] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                stride_tensor,
            )
            masks = batch["masks"].to(self.device).float()
            if tuple(masks.shape[-2:]) != (mask_h, mask_w):
                masks = F.interpolate(masks[None], (mask_h, mask_w), mode="nearest")[0]

            loss[1] = self.calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz, self.overlap
            )
        else:
            loss[1] += (proto * 0).sum() + (pred_masks * 0).sum()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.box
        loss[2] *= self.hyp.cls
        loss[3] *= self.hyp.dfl

        return loss.sum() * batch_size, loss.detach()

    @staticmethod
    def single_mask_loss(gt_mask, pred, proto, xyxy, area):
        pred_mask = torch.einsum("in,nhw->ihw", pred, proto)
        loss = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
        return (crop_mask(loss, xyxy).mean(dim=(1, 2)) / area).sum()

    def calculate_segmentation_loss(self, fg_mask, masks, target_gt_idx, target_bboxes,
                                    batch_idx, proto, pred_masks, imgsz, overlap):
        _, _, mask_h, mask_w = proto.shape
        loss = 0

        target_bboxes_normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(target_bboxes_normalized)[..., 2:].prod(2)
        mxyxy = target_bboxes_normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)

        for i, single_i in enumerate(zip(fg_mask, target_gt_idx, pred_masks, proto, mxyxy, marea, masks)):
            fg_mask_i, target_gt_idx_i, pred_masks_i, proto_i, mxyxy_i, marea_i, masks_i = single_i
            if fg_mask_i.any():
                mask_idx = target_gt_idx_i[fg_mask_i]
                if overlap:
                    gt_mask = masks_i == (mask_idx + 1).view(-1, 1, 1)
                    gt_mask = gt_mask.float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]

                loss += self.single_mask_loss(
                    gt_mask, pred_masks_i[fg_mask_i], proto_i, mxyxy_i[fg_mask_i], marea_i[fg_mask_i]
                )
            else:
                loss += (proto * 0).sum() + (pred_masks * 0).sum()

        # [FIX-5] was: loss / fg_mask.sum  (bound method, crashed when seg used)
        return loss / fg_mask.sum()


class DetectAuxLoss:
    """Train-only auxiliary-head deep-supervision loss.

    Round 20 -- DetectAux adds a parallel detection head over the same feature
    maps as the main head; it is supervised during training and DROPPED at
    inference (zero deploy cost). The total loss is the main detection loss plus
    a down-weighted auxiliary detection loss, giving the shared neck features an
    extra gradient signal. Both heads share strides, so the same v8DetectionLoss
    is reused for each. Mirrors E2EDetectLoss's two-loss structure.
    """

    def __init__(self, model, aux_weight=0.25):
        """Initialize with a shared detection loss and the auxiliary weight.

        Reads aux_weight from the DetectAux head (set via YAML) when present.
        """
        self.det = v8DetectionLoss(model, tal_topk=10)
        self.aux_weight = getattr(model.model[-1], "aux_weight", aux_weight)

    def __call__(self, preds, batch):
        """Main loss + aux_weight * auxiliary loss; logs the main head's items.

        Training: preds is {"main", "aux"} -> supervise both. Validation: the
        model is in eval mode and returns only the main feats (no aux), so fall
        back to the plain detection loss.
        """
        preds = preds[1] if isinstance(preds, tuple) else preds
        if not isinstance(preds, dict):  # val/eval path: only main head present
            return self.det(preds, batch)
        loss_main = self.det(preds["main"], batch)
        loss_aux = self.det(preds["aux"], batch)
        return loss_main[0] + self.aux_weight * loss_aux[0], loss_main[1]


class DetectObjLoss(v8DetectionLoss):
    """v8 detection loss + an objectness (foreground/background) BCE term.

    Round 24 -- supervises DetectObj's per-anchor objectness logit against the
    TAL foreground mask (1 = assigned foreground, 0 = background).

    [NEW-7] stride_tensor is now passed to bbox_loss (pixel-space weighting was
    previously silently in the grid-units fallback) and the center loss is
    applied, so this loss differs from the base v8DetectionLoss ONLY by the
    objectness term -- required for a clean ablation comparison.
    """

    def __init__(self, model, obj_weight=1.0):
        """Initialize the base detection loss plus an objectness BCE."""
        super().__init__(model)
        self.obj_weight = obj_weight
        self.bce_obj = nn.BCEWithLogitsLoss(reduction="none")

    def __call__(self, preds, batch):
        """Detection loss on the main head + objectness BCE on the obj branch."""
        preds = preds[1] if isinstance(preds, tuple) else preds
        if not isinstance(preds, dict):  # val/eval: only main head present
            return super().__call__(preds, batch)

        # epoch sync (mirror of base __call__)
        try:
            if hasattr(self._model, 'current_epoch'):
                self.epoch = self._model.current_epoch
        except:
            pass
        self._sync_bbox_loss_state()

        feats, obj_feats = preds["main"], preds["obj"]
        loss = torch.zeros(4, device=self.device)  # box, cls, dfl, obj
        pred_distri, pred_scores = torch.cat(
            [xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2
        ).split((self.reg_max * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()
        pred_obj = torch.cat(
            [oi.view(feats[0].shape[0], 1, -1) for oi in obj_feats], 2
        ).permute(0, 2, 1).contiguous()  # (b, A, 1)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, _ = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_scores_sum = max(target_scores.sum(), 1)
        # [NEW-4] cls loss via the shared helper (BCE or VFL)
        loss[1] = self._compute_cls_loss(pred_scores, target_scores, target_scores_sum, dtype)

        center_loss = torch.tensor(0.0, device=self.device)
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            # [NEW-7] pass stride_tensor -> pixel-space weighting active
            loss[0], loss[2] = self.bbox_loss(
                pred_distri, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, stride_tensor
            )
            # [NEW-7] center loss applied, mirroring the base loss
            center_loss = self._compute_center_loss(
                pred_bboxes, target_bboxes, fg_mask, stride_tensor
            )

        # objectness BCE: target = foreground mask
        obj_target = fg_mask.unsqueeze(-1).to(dtype)  # (b, A, 1)
        loss[3] = self.bce_obj(pred_obj, obj_target).mean()

        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl
        loss[3] *= self.obj_weight
        # [NEW-8] center loss after box gain
        loss[0] = loss[0] + center_loss
        return loss.sum() * batch_size, loss[:3].detach()  # log box/cls/dfl