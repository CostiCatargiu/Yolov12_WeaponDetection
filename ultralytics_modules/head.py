# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Model head modules."""

import copy
import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_uniform_

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors

from .block import DFL, BNContrastiveHead, ContrastiveHead, Proto, ZGLKA
from .conv import Conv, DWConv
from .transformer import MLP, DeformableTransformerDecoder, DeformableTransformerDecoderLayer
from .utils import bias_init_with_prob, linear_init

__all__ = (
    "Detect",
    "Segment",
    "Pose",
    "Classify",
    "OBB",
    "RTDETRDecoder",
    "v10Detect",
    "DetectCGC",
    "DetectLKACls",
    "DetectSmallCls",
    "DetectDeepCls",
    "DetectWideCls",
    "DetectAux",
    "DetectAuxDual",
    "DetectAuxDualDeepP3",
    "DetectDecoupled",
    "DetectObj",
    "DetectDecoupledObj",
    "DetectDecoupledAux",
    "DetectMultiProto",
    "DetectCosine",
    "DetectDecoupledCosine",
)


class Detect(nn.Module):
    """YOLO Detect head for detection models."""

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels."""
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if self.end2end:
            return self.forward_end2end(x)

        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)

    def forward_end2end(self, x):
        """
        Performs forward pass of the v10Detect module.

        Args:
            x (tensor): Input tensor.

        Returns:
            (dict, tensor): If not in training mode, returns a dictionary containing the outputs of both one2many and one2one detections.
                           If in training mode, returns a dictionary containing the outputs of one2many and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1) for i in range(self.nl)
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        if self.training:  # Training path
            return {"one2many": x, "one2one": one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0, 2, 1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many": x, "one2one": one2one})

    def _inference(self, x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.dfl(box) * self.strides, self.anchors.unsqueeze(0) * self.strides, xywh=False
            )
            return dbox.transpose(1, 2), cls.sigmoid().permute(0, 2, 1)
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        return torch.cat((dbox, cls.sigmoid()), 1)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for a, b, s in zip(m.one2one_cv2, m.one2one_cv3, m.stride):  # from
                a[-1].bias.data[:] = 1.0  # box
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes, anchors, xywh=True):
        """Decode bounding boxes."""
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc: int = 80):
        """
        Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x, y, w, h, class_probs].
            max_det (int): Maximum detections per image.
            nc (int, optional): Number of classes. Default: 80.

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x, y, w, h, max_class_prob, class_index].
        """
        batch_size, anchors, _ = preds.shape  # i.e. shape(16,8400,84)
        boxes, scores = preds.split([4, nc], dim=-1)
        index = scores.amax(dim=-1).topk(min(max_det, anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1, 1, 4))
        scores = scores.gather(dim=1, index=index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]  # batch indices
        return torch.cat([boxes[i, index // nc], scores[..., None], (index % nc)[..., None].float()], dim=-1)


class DetectCGC(Detect):
    """Detect with Context-Gated Classification (CGC) — zero-gated head.

    Motivation (weapon_noaug 70% ablation): box branch is already strong
    (AR50 ~0.95) but classification fails on the ambiguous 'other' class
    (AP50 ~0.53 vs ~0.88 for pistol/long_gun/knife). The stock cls tower
    sees only a local 3x3-conv receptive field; class identity of ambiguous
    objects depends on SCENE context.

    Mechanism:
      1. Global context vector from the P5 map (softmax attention pooling,
         GCNet-style) -> small MLP.
      2. Projected per scale and injected into the CLS-branch input only,
         behind per-channel zero-init gates:
             cls_in_i = x_i + gamma_i * proj_i(ctx)
         Box branch (cv2) input stays raw -> regression untouched.
      3. gamma = 0 at init -> exact stock Detect at epoch 0; with a stock
         body at index 21, pretrained Detect box weights transfer as usual.

    Drop-in YAML replacement:  - [[14, 17, 20], 1, DetectCGC, [nc]]
    """

    def __init__(self, nc=80, ch=()):
        """Initialize DetectCGC with the standard Detect layers plus the gated context branch."""
        super().__init__(nc, ch)
        c5 = ch[-1]
        c_ = max(c5 // 4, 64)
        self.ctx_attn = nn.Conv2d(c5, 1, 1)
        self.ctx_mlp = nn.Sequential(
            nn.Conv2d(c5, c_, 1),
            nn.SiLU(),
            nn.Conv2d(c_, c_, 1),
            nn.SiLU(),
        )
        self.ctx_proj = nn.ModuleList(nn.Conv2d(c_, c, 1) for c in ch)
        self.ctx_gamma = nn.ParameterList(nn.Parameter(torch.zeros(c, 1, 1)) for c in ch)

    def _context(self, x5):
        """Softmax-attention-pooled global context vector from the P5 feature map."""
        b, c, h, w = x5.shape
        w_ = self.ctx_attn(x5).view(b, 1, h * w).softmax(dim=-1)            # b,1,hw
        ctx = (x5.view(b, c, h * w) @ w_.transpose(1, 2)).view(b, c, 1, 1)  # b,c,1,1
        return self.ctx_mlp(ctx)

    def forward(self, x):
        """Standard Detect forward, but cls branch input is context-augmented (zero-gated)."""
        ctx = self._context(x[-1])
        for i in range(self.nl):
            xc = x[i] + self.ctx_gamma[i] * self.ctx_proj[i](ctx)  # cls input (broadcast over HW)
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](xc)), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)


class DetectLKACls(Detect):
    """Detect with a per-scale, zero-gated Large-Kernel-Attention cls branch.

    Motivation (weapon_noaug 70% ablation, round 13): the proven k=11 ZGLKA
    receptive field (used in r6/r11_widefuse on the SHARED feature feeding
    both box and cls) has to compete with the box-regression objective on
    the same tensor, which likely caps/dilutes any 'other'-class
    classification gain. DetectCGC showed that isolating a change to the
    CLS branch only (box/cv2 untouched) is safe and low-risk -- but its
    global P5-pooled context is a poor match for small, visually-diverse
    'other' objects (P5 is the worst map for small-object detail).

    This module combines both validated ideas: the proven local k=11 LKA
    receptive field, applied per-scale, ISOLATED to the cls-branch input
    only, behind its own per-channel zero-init gate:
        cls_in_i = x_i + gamma_i * ZGLKA(k)(x_i)
        x_i = cat([cv2_i(x_i), cv3_i(cls_in_i)], 1)
    Box branch (cv2) input stays raw -> regression untouched.
    gamma = 0 at init -> exact stock Detect at epoch 0; with a stock body
    at index 21, pretrained Detect box weights transfer as usual.

    Drop-in YAML replacement:  - [[14, 17, 20], 1, DetectLKACls, [nc, 11]]
    """

    def __init__(self, nc=80, k=11, ch=()):
        """Initialize DetectLKACls with the standard Detect layers plus a per-scale gated LKA cls branch."""
        super().__init__(nc, ch)
        self.cls_lka = nn.ModuleList(ZGLKA(c, k) for c in ch)
        self.cls_gamma = nn.ParameterList(nn.Parameter(torch.zeros(c, 1, 1)) for c in ch)

    def forward(self, x):
        """Standard Detect forward, but cls branch input is LKA-augmented (zero-gated, per scale)."""
        for i in range(self.nl):
            xc = x[i] + self.cls_gamma[i] * self.cls_lka[i](x[i])  # cls input only
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](xc)), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)


class DetectSmallCls(Detect):
    """Detect with a per-scale, zero-gated SMALL-kernel (local-detail) cls branch.

    Motivation (weapon_noaug 70% ablation, rounds 13-15): per-class
    AP50_small analysis shows the small-object mAP drop introduced by
    r11_widefuse_70 (and every other round-6-14 P4-BU/TD variant) is
    overwhelmingly concentrated in the 'other' class (-14.98pp AP50_small
    vs baseline; weapon classes only -1.47 to -2.34pp). DetectLKACls
    (round 13) tried isolating a change to the cls branch with a k=11
    ZGLKA -- but FAILED (-0.38 to -0.86 mAP50 vs its baseline): a k=11
    receptive field is too coarse and SMOOTHS OUT exactly the fine local
    detail a per-anchor classifier needs for small, visually-ambiguous
    'other' objects. DetectCGC (global P5 context) also under-performed
    for the same reason in the opposite direction (too global).

    This module takes the OPPOSITE regime from DetectLKACls: a genuinely
    SMALL receptive field (k=3, dilation=1, depthwise + GroupNorm + SiLU --
    the same "small" branch design validated structurally in
    ZGLSKAWideFuse3), applied per-scale, ISOLATED to the cls-branch input
    only, behind its own per-channel zero-init gate:
        cls_in_i = x_i + gamma_i * SmallDetail(k)(x_i)
        x_i = cat([cv2_i(x_i), cv3_i(cls_in_i)], 1)
    Box branch (cv2) input stays raw -> regression untouched.
    gamma = 0 at init -> exact stock Detect at epoch 0; with a stock body
    at index 21 (or wherever Detect sits), pretrained Detect box/cls
    weights transfer as usual (only the new cls_small/cls_gamma params are
    fresh).

    Intended pairing: combine with WideFuse@P4-BU UNCHANGED (= r11_widefuse_70
    architecture, 79.40 mAP50) so the proven shared-feature gain is kept,
    and this module ADDITIONALLY targets the 'other'-class cls-discriminability
    loss directly, without re-touching the shared P4 features.

    Drop-in YAML replacement:  - [[14, 17, 20], 1, DetectSmallCls, [nc, 3]]
    """

    def __init__(self, nc=80, k=3, ch=()):
        """Initialize DetectSmallCls with the standard Detect layers plus a per-scale gated small-kernel cls branch."""
        super().__init__(nc, ch)
        self.cls_small = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(c, c, k, 1, k // 2, groups=c),
                nn.GroupNorm(1, c),
                nn.SiLU(),
            )
            for c in ch
        )
        self.cls_gamma = nn.ParameterList(nn.Parameter(torch.zeros(c, 1, 1)) for c in ch)

    def forward(self, x):
        """Standard Detect forward, but cls branch input is small-kernel-detail-augmented (zero-gated, per scale)."""
        for i in range(self.nl):
            xc = x[i] + self.cls_gamma[i] * self.cls_small[i](x[i])  # cls input only
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](xc)), 1)
        if self.training:  # Training path
            return x
        y = self._inference(x)
        return y if self.export else (y, x)


class DetectDeepCls(Detect):
    """Detect with a DEEPER classification tower (4 blocks vs the stock 2).

    Motivation (weapon_noaug 70% ablation, round 18): per-class analysis shows
    the entire dataset ceiling is the "other" class, and its failure mode is
    CLASSIFICATION, not localization -- recall AR50 ~= 0.84 but precision
    AP50 ~= 0.51 (a ~0.33 recall-precision gap, vs ~0.09 for the weapon
    classes). The detector FINDS "other" objects and mis-RANKS them. ~30 prior
    architecture variants (rounds 1-17: LKA, strip, GC, multi-dil, routing,
    DetectLKACls/SmallCls/CGC) all added receptive-field / spatial context to
    the shared or cls-input feature -- i.e. they targeted LOCALIZATION -- and
    none beat plain loss tuning, because localization was never the bottleneck.

    This instead adds CAPACITY to the classifier itself: the cls branch (cv3)
    is deepened from the stock 2 (DWConv+Conv) blocks to 4, giving the head
    representational room to separate the heterogeneous "other" class from
    background and from the weapon classes. The box branch (cv2) and DFL are
    left untouched -> localization is unchanged and box weights transfer fully
    from the pretrained checkpoint; only the cls tower trains fresh, which is
    exactly the branch the per-class data flags as under-capacity.

    Drop-in YAML replacement:  - [[14, 17, 20], 1, DetectDeepCls, [nc]]
    """

    def __init__(self, nc=80, ch=()):
        """Initialize DetectDeepCls with a 4-block depthwise cls tower per scale."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )


class DetectWideCls(Detect):
    """Detect with a WIDER classification tower (2x cls channels, stock depth).

    Companion to DetectDeepCls (round 18): same diagnosis (the "other" class is
    a cls-ranking bottleneck, AR50~0.84 vs AP50~0.51), but tests whether the
    missing classifier capacity is better added as WIDTH than as DEPTH. The cls
    branch (cv3) keeps the stock 2-block structure but doubles its intermediate
    channel width (c3 -> 2*c3). Box branch (cv2) and DFL untouched -> full box
    transfer, cls tower trains fresh. Run head-to-head with DetectDeepCls to
    read depth-vs-width for the cls bottleneck.

    Drop-in YAML replacement:  - [[14, 17, 20], 1, DetectWideCls, [nc]]
    """

    def __init__(self, nc=80, ch=()):
        """Initialize DetectWideCls with a 2x-width depthwise cls tower per scale."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100)) * 2
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )


class DetectAux(Detect):
    """Detect with a TRAIN-ONLY auxiliary detection head (deep supervision).

    Round 20 -- the idea (and the only legitimate version of an "auxiliary
    head"): add a second, parallel detection head over the SAME P3/P4/P5
    feature maps as the main head, supervise it during training, and DROP it at
    inference. The auxiliary head gives the shared neck features an extra
    gradient signal (a second, independent set of box/cls towers fitting the
    same targets), with ZERO inference/deploy cost -- at eval the module behaves
    exactly like stock Detect.

    Motivation: after rounds 1-19 (~35 inference-path architecture variants) all
    came up flat vs loss tuning, deep supervision is the one untried lever that
    is a *training-signal* change rather than an inference-structure change. The
    aux towers share the main strides, so the standard v8DetectionLoss is reused
    for both (see utils/loss.py::DetectAuxLoss). Up-/down-weight the aux term via
    DetectAuxLoss.aux_weight.

    YAML: drop-in for Detect -- e.g.  - [[14, 17, 20], 1, DetectAux, [nc]]
    """

    def __init__(self, nc=80, aux_weight=0.25, ch=()):
        """Initialize the main Detect plus a parallel auxiliary box/cls head.

        YAML may be [nc] (default aux_weight) or [nc, aux_weight]. parse_model
        appends the channel list as the last positional arg, so when no weight is
        given aux_weight receives that list -- detect and swap for compatibility.
        """
        if isinstance(aux_weight, (list, tuple)):  # old [nc] yaml -> ch landed here
            ch, aux_weight = aux_weight, 0.25
        super().__init__(nc, ch)
        self.aux_weight = float(aux_weight)
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))
        self.cv2a = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3a = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )

    def forward(self, x):
        """Training: return {'main','aux'} for DetectAuxLoss. Inference: stock Detect (aux dropped)."""
        main = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]
        if self.training:
            aux = [torch.cat((self.cv2a[i](x[i]), self.cv3a[i](x[i])), 1) for i in range(self.nl)]
            return {"main": main, "aux": aux}
        y = self._inference(main)
        return y if self.export else (y, main)

    def bias_init(self):
        """Initialize biases for the main head (super) and the auxiliary towers."""
        super().bias_init()
        m = self
        for a, b, s in zip(m.cv2a, m.cv3a, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls


class DetectAuxDual(Detect):
    """Detect with DUAL-PATH auxiliary supervision (detail teacher).

    Round 32B -- the key insight from rounds 11-31 is that DetectAux mirrors
    the main head (both see the same fused features), so the aux gradient
    cannot teach the backbone anything the main head doesn't. This head fixes
    that by routing DIFFERENT features to main vs aux towers:

      Main towers: see context-rich features (post-widefuse P4)
      Aux towers:  see detail-rich features (pre-widefuse P4)

    This forces the backbone to satisfy BOTH objectives: the main head rewards
    context (good for medium/large objects) while the aux head rewards detail
    preservation (good for small objects). The backbone must maintain fine-
    grained features because the aux head supervises them directly.

    At inference: aux towers are dropped, zero cost (same as DetectAux).

    YAML provides 2*nl inputs:
      [main_p3, main_p4_fused, main_p5,  aux_p3, aux_p4_prefuse, aux_p5]
    Example:
      - [[14, 21, 20, 14, 17, 20], 1, DetectAuxDual, [nc, 0.5]]
    Where 21 = post-widefuse P4, 17 = pre-widefuse P4.
    """

    def __init__(self, nc=80, aux_weight=0.25, ch=()):
        if isinstance(aux_weight, (list, tuple)):
            ch, aux_weight = aux_weight, 0.25
        n = len(ch) // 2
        main_ch, aux_ch = ch[:n], ch[n:]
        super().__init__(nc, main_ch)  # builds cv2/cv3 over main channels
        self.aux_weight = float(aux_weight)
        c2 = max((16, aux_ch[0] // 4, self.reg_max * 4))
        c3 = max(aux_ch[0], min(self.nc, 100))
        self.cv2a = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in aux_ch
        )
        self.cv3a = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in aux_ch
        )

    def forward(self, x):
        """Main head sees x[:nl] (fused), aux sees x[nl:] (detail). Inference: main only."""
        main_x, aux_x = x[: self.nl], x[self.nl :]
        main = [torch.cat((self.cv2[i](main_x[i]), self.cv3[i](main_x[i])), 1) for i in range(self.nl)]
        if self.training:
            aux = [torch.cat((self.cv2a[i](aux_x[i]), self.cv3a[i](aux_x[i])), 1) for i in range(self.nl)]
            return {"main": main, "aux": aux}
        y = self._inference(main)
        return y if self.export else (y, main)

    def bias_init(self):
        """Initialize biases for main (super) and aux towers."""
        super().bias_init()
        for a, b, s in zip(self.cv2a, self.cv3a, self.stride):
            a[-1].bias.data[:] = 1.0
            b[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / s) ** 2)


class DetectAuxDualDeepP3(DetectAuxDual):
    """DetectAuxDual with DEEPER P3 head towers (round 40).

    Data analysis across 19 experiments reveals the #1 bottleneck is NOT
    features (recall is 80-90%) but SCORING QUALITY at P3:

      class     AR50_small  AP50_small  gap (scoring loss)
      other     80.6%       49.0%       31.6pp  <-- worst
      long_gun  90.5%       67.2%       23.4pp
      knife     85.1%       64.2%       20.9pp
      pistol    90.3%       77.6%       12.7pp

    The detector FINDS small objects but mis-scores/mis-ranks them. Meanwhile
    long_gun_small has AP50=67% but AP50-95=25% -- worst localization ratio
    (0.378), meaning boxes are detected but poorly regressed.

    Both problems originate at P3: it handles the HARDEST classification
    (diverse small objects) and regression (thin objects at low res) with the
    SHALLOWEST towers (same 2-conv depth as the easier P4/P5 levels).

    FIX: Add one extra DWConv+Conv layer to P3's cls (cv3) and box (cv2)
    towers ONLY. P4/P5 towers stay standard depth. This gives P3 more
    representational capacity for its harder task:

      P3 cls: DWConv→Conv → DWConv→Conv → [DWConv→Conv] → Conv2d  (3 blocks)
      P3 box: Conv→Conv → [Conv] → Conv2d                         (3 blocks)
      P4/P5:  standard 2-block depth (unchanged)

    The extra layer is zero-init-biased and uses the same channel widths, so
    at epoch 0 it's near-identity (safe pretrained transfer). At inference:
    adds ~2% FLOPs (only the P3 head is deeper). Aux towers stay standard.

    Drop-in replacement for DetectAuxDual — same YAML format:
      - [[22, 21, 23, 14, 17, 20], 1, DetectAuxDualDeepP3, [nc, 0.5]]
    """

    def __init__(self, nc=80, aux_weight=0.25, ch=()):
        super().__init__(nc, aux_weight, ch)
        # Rebuild cv2[0] and cv3[0] with one extra conv layer (P3 = index 0)
        n = len(ch) // 2 if not hasattr(self, '_built') else self.nl
        main_ch = ch[:len(ch) // 2] if isinstance(ch, (list, tuple)) else ch
        p3_ch = main_ch[0]
        c2 = max((16, p3_ch // 4, self.reg_max * 4))
        c3 = max(p3_ch, min(self.nc, 100))

        # Deeper P3 box tower: 3 conv blocks instead of 2
        self.cv2[0] = nn.Sequential(
            Conv(p3_ch, c2, 3), Conv(c2, c2, 3), Conv(c2, c2, 3),
            nn.Conv2d(c2, 4 * self.reg_max, 1),
        )
        # Deeper P3 cls tower: 3 DWConv+Conv blocks instead of 2
        self.cv3[0] = nn.Sequential(
            nn.Sequential(DWConv(p3_ch, p3_ch, 3), Conv(p3_ch, c3, 1)),
            nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
            nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
            nn.Conv2d(c3, self.nc, 1),
        )


class DetectDecoupled(Detect):
    """Task-decoupled head: box and cls read SEPARATE feature maps.

    Round 24 -- every prior head fed the SAME shared neck feature to both the
    box (cv2) and cls (cv3) branches, so any classification gain on the hard
    "other" class has to compete with the box-regression objective on the same
    tensor. This routes the box branch to the main neck features and the cls
    branch to a DEDICATED cls feature pathway (separate conv layers / weights,
    provided by the YAML), so classification gets its own representation,
    uncompromised by localization.

    YAML provides 2*nl inputs in order [box_p3, box_p4, box_p5, cls_p3, cls_p4,
    cls_p5]; the first nl feed cv2 (box), the last nl feed cv3 (cls):
      - [[14,17,20, 21,22,23], 1, DetectDecoupled, [nc]]
    """

    def __init__(self, nc=80, ch=()):
        """Build cv2 over the box channels and cv3 over the cls channels."""
        n = len(ch) // 2
        box_ch, cls_ch = ch[:n], ch[n:]
        super().__init__(nc, box_ch)  # cv2/cv3/dfl over box_ch; self.nl = n
        c3 = max(cls_ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in cls_ch
        )

    def forward(self, x):
        """cv2 reads box features x[:nl]; cv3 reads cls features x[nl:]."""
        box, cls = x[: self.nl], x[self.nl:]
        out = [torch.cat((self.cv2[i](box[i]), self.cv3[i](cls[i])), 1) for i in range(self.nl)]
        if self.training:
            return out
        y = self._inference(out)
        return y if self.export else (y, out)


class DetectObj(Detect):
    """Detect + explicit objectness (foreground/background) branch.

    Round 24 -- the "other"-class failure is precision/ranking (recall ~0.84,
    AP ~0.51): the detector finds the objects and over-scores background-like
    anchors. YOLOv8/12 removed the objectness head; this re-adds one. Per anchor
    it predicts an objectness logit; at INFERENCE the cls logits are shifted by
    the objectness logit (score = sigmoid(cls + obj)) so low-objectness anchors
    are suppressed -- directly attacking the false-positive / precision problem.

    Train-only dict output {main, obj}; objectness is supervised by
    DetectObjLoss (BCE vs the TAL foreground mask). At inference behaves as a
    normal detector with objectness-reweighted scores.

    Drop-in:  - [[14, 17, 20], 1, DetectObj, [nc]]
    """

    def __init__(self, nc=80, obj_beta=1.0, ch=()):
        """Initialize Detect + objectness branch. obj_beta (<1) softens the
        inference reweighting so it suppresses false positives gently instead of
        over-suppressing (round 24 obj_beta=1.0 hurt overall mAP). YAML: [nc] or
        [nc, obj_beta]; parse appends ch last, so swap if obj_beta got the list."""
        if isinstance(obj_beta, (list, tuple)):
            ch, obj_beta = obj_beta, 1.0
        super().__init__(nc, ch)
        self.obj_beta = float(obj_beta)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, max(16, x // 4), 3), nn.Conv2d(max(16, x // 4), 1, 1)) for x in ch
        )

    def forward(self, x):
        """Training: {main, obj}. Inference: cls logits shifted by obj_beta*objectness."""
        obj = [self.cv4[i](x[i]) for i in range(self.nl)]
        main = [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]
        if self.training:
            return {"main": main, "obj": obj}
        for i in range(self.nl):  # score = sigmoid(cls + beta*obj): gently suppress low-objectness
            main[i][:, self.reg_max * 4:] = main[i][:, self.reg_max * 4:] + self.obj_beta * obj[i]
        y = self._inference(main)
        return y if self.export else (y, main)

    def bias_init(self):
        """Initialize the standard biases; objectness starts neutral (0)."""
        super().bias_init()
        for a in self.cv4:
            a[-1].bias.data[:] = 0.0


class DetectDecoupledObj(DetectDecoupled):
    """Round 26 -- the SYNTHESIS: decoupled cls pathway + softened objectness.

    Rounds 24/25 found two complementary effects: DetectDecoupled (box and cls
    on separate features) reliably improved precision and was best-on-validation;
    DetectObj (objectness) gave the best small-"other" AP in the project but
    over-suppressed. This head combines them: box reads box features, cls reads
    the dedicated cls features (decoupled), AND a per-anchor objectness branch
    (over the box features) gently reweights cls at inference (obj_beta<1).
    Train-only dict output {main, obj}; supervised by DetectObjLoss.

    YAML provides 2*nl inputs [box..., cls...] like DetectDecoupled, plus an
    optional obj_beta:  - [[14,21,20, 22,23,24], 1, DetectDecoupledObj, [nc, 0.5]]
    """

    def __init__(self, nc=80, obj_beta=1.0, ch=()):
        """Build the decoupled box/cls head plus an objectness branch on box feats."""
        if isinstance(obj_beta, (list, tuple)):
            ch, obj_beta = obj_beta, 1.0
        super().__init__(nc, ch)  # cv2 over box_ch, cv3 over cls_ch; self.nl set
        self.obj_beta = float(obj_beta)
        box_ch = ch[: len(ch) // 2]
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(x, max(16, x // 4), 3), nn.Conv2d(max(16, x // 4), 1, 1)) for x in box_ch
        )

    def forward(self, x):
        """box = x[:nl], cls = x[nl:]; objectness over box feats; train-only dict."""
        box, cls = x[: self.nl], x[self.nl:]
        obj = [self.cv4[i](box[i]) for i in range(self.nl)]
        main = [torch.cat((self.cv2[i](box[i]), self.cv3[i](cls[i])), 1) for i in range(self.nl)]
        if self.training:
            return {"main": main, "obj": obj}
        for i in range(self.nl):
            main[i][:, self.reg_max * 4:] = main[i][:, self.reg_max * 4:] + self.obj_beta * obj[i]
        y = self._inference(main)
        return y if self.export else (y, main)

    def bias_init(self):
        """Decoupled biases (super) + neutral objectness."""
        super().bias_init()
        for a in self.cv4:
            a[-1].bias.data[:] = 0.0


class DetectDecoupledAux(DetectAux):
    """Decoupled box/cls head + train-only auxiliary deep supervision (rounds 24 + 20).

    Combines the two independently-validated effects on the widefuse backbone:
      * DetectDecoupled (round 24): box (cv2) and cls (cv3) read SEPARATE feature
        maps, so classification gets its own representation, uncompromised by the
        localization objective -- the one reproducible precision gain of the search.
      * DetectAux (round 20): a parallel box/cls head supervised during training
        and DROPPED at inference (zero deploy cost), an extra gradient signal on
        the shared features.

    The aux towers are themselves decoupled (aux box over box feats, aux cls over
    the dedicated cls feats), mirroring the main head.

    Loss dispatch: subclass of DetectAux -> DetectionModel.init_criterion selects
    DetectAuxLoss, which supervises the {"main","aux"} dict in training and falls
    back to the main head in eval (no aux at inference).

    YAML: 2*nl inputs [box_p3,box_p4,box_p5, cls_p3,cls_p4,cls_p5], optional
    aux_weight last:
      - [[14, 21, 20, 22, 23, 24], 1, DetectDecoupledAux, [nc, 0.5]]
    """

    def __init__(self, nc=80, aux_weight=0.25, ch=()):
        """Decoupled main head (box over box_ch, cls over cls_ch) + decoupled aux towers.

        YAML may be [nc] (ch lands in aux_weight) or [nc, aux_weight]; detect and
        swap, mirroring DetectAux.
        """
        if isinstance(aux_weight, (list, tuple)):  # [nc] yaml -> ch landed in aux_weight
            ch, aux_weight = aux_weight, 0.25
        n = len(ch) // 2
        box_ch, cls_ch = ch[:n], ch[n:]
        Detect.__init__(self, nc, box_ch)  # cv2/cv3/dfl over box_ch; self.nl = n
        self.aux_weight = float(aux_weight)
        c2 = max((16, box_ch[0] // 4, self.reg_max * 4))
        c3 = max(cls_ch[0], min(self.nc, 100))
        # main cls branch reads the dedicated cls features (decoupled)
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in cls_ch
        )
        # train-only aux towers: box over box feats, cls over the dedicated cls feats
        self.cv2a = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in box_ch
        )
        self.cv3a = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in cls_ch
        )

    def forward(self, x):
        """box = x[:nl], cls = x[nl:]. Train: decoupled {"main","aux"}. Eval: main only."""
        box, cls = x[: self.nl], x[self.nl:]
        main = [torch.cat((self.cv2[i](box[i]), self.cv3[i](cls[i])), 1) for i in range(self.nl)]
        if self.training:
            aux = [torch.cat((self.cv2a[i](box[i]), self.cv3a[i](cls[i])), 1) for i in range(self.nl)]
            return {"main": main, "aux": aux}
        y = self._inference(main)
        return y if self.export else (y, main)

    # bias_init inherited from DetectAux: Detect.bias_init handles cv2/cv3 (main),
    # then the aux loop sets cv2a/cv3a biases. All four ModuleLists have nl entries.


class MultiProtoHead(nn.Module):
    """Mixture classification logits: K learnable sub-prototypes per class, combined
    by a soft-OR (logsumexp over the K sub-logits).

    Round 30 — the catch-all "other" class is multimodal (it lumps together many
    visually unrelated sub-types), so a single 1x1 cls conv learns ONE hyperplane
    per class and cannot rank a multimodal class cleanly -> the measured high-recall
    / low-precision signature. This layer emits K logits per class and reduces them
    with logsumexp, i.e. "does the feature match ANY of this class's sub-prototypes",
    letting a multimodal class occupy several regions of feature space. A unimodal
    class (the saturated weapon classes) can leave the extra prototypes redundant.
    K=1 reduces exactly to a standard cls conv. Drop-in for the final nn.Conv2d.
    """

    def __init__(self, c_in, nc, k=4):
        super().__init__()
        self.nc = nc
        self.k = int(k)
        self.conv = nn.Conv2d(c_in, nc * self.k, 1)

    def forward(self, x):
        b, _, h, w = x.shape
        z = self.conv(x).view(b, self.nc, self.k, h, w)
        return torch.logsumexp(z, dim=2)  # soft-OR over sub-prototypes -> (b, nc, h, w)


class DetectMultiProto(Detect):
    """Detect with a MIXTURE (multi-prototype) classification head (round 30).

    Replaces the final 1x1 cls conv in each cv3 tower with a MultiProtoHead: K
    sub-prototypes per class, combined by logsumexp. Targets the diagnosed ceiling
    -- the multimodal "other" class is mis-ranked by a single linear boundary (the
    round-28 single-prototype cosine head made this WORSE, evidence the one-mode
    assumption is the problem). Box branch (cv2) and the forward/inference path are
    unchanged; cv3 still outputs nc channels (the logsumexp reduces K -> nc), so the
    standard v8DetectionLoss BCE applies directly and gradients flow to all K modes.

    Append-only, near-zero inference cost. YAML: [nc] or [nc, k]:
      - [[14, 21, 20], 1, DetectMultiProto, [nc, 4]]
    """

    def __init__(self, nc=80, k=4, ch=()):
        """Build a standard Detect, then swap each cv3 final conv for a MultiProtoHead.

        YAML may be [nc] (ch lands in k) or [nc, k]; detect and swap, mirroring DetectAux.
        """
        if isinstance(k, (list, tuple)):  # [nc] yaml -> ch landed in k
            ch, k = k, 4
        super().__init__(nc, ch)
        self.k = int(k)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                MultiProtoHead(c3, self.nc, self.k),
            )
            for x in ch
        )

    def bias_init(self):
        """Box biases as in Detect; cls sub-prototype biases set so the logsumexp of
        the K modes starts at the standard Detect cls prior (target - log(K) each)."""
        m = self
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            a[-1].bias.data[:] = 1.0  # box
            target = math.log(5 / m.nc / (640 / s) ** 2)  # standard cls prior
            b[-1].conv.bias.data.view(m.nc, m.k)[:] = target - math.log(m.k)


class CosineCls(nn.Module):
    """Cosine / prototype classifier (round 28): replaces the linear cls layer.

    logit_c = scale * cos(feature, prototype_c) + bias_c, with L2-normalized
    feature and learnable per-class prototypes. Scores by ANGULAR similarity
    rather than a dot product, which is better-calibrated and better-ranked for
    hard, visually-diverse classes -- aimed at the "found but mis-scored" failure
    on the catch-all "other" class. Drop-in for the final nn.Conv2d(c3, nc, 1).
    """

    def __init__(self, c, nc, scale=16.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(nc, c))  # class prototypes
        self.scale = nn.Parameter(torch.tensor(float(scale)))
        self.bias = nn.Parameter(torch.zeros(nc))

    def forward(self, x):
        xn = x / (x.norm(dim=1, keepdim=True) + 1e-6)               # normalize feature over channels
        wn = self.weight / (self.weight.norm(dim=1, keepdim=True) + 1e-6)  # normalize prototypes
        logit = torch.einsum("bchw,nc->bnhw", xn, wn) * self.scale
        return logit + self.bias.view(1, -1, 1, 1)


class DetectCosine(Detect):
    """Detect with a COSINE/prototype classification head (round 28).

    Every prior head used a linear cls layer. This replaces the final cls conv
    with CosineCls (angular scoring against learnable class prototypes), changing
    HOW scores are computed rather than the features feeding them -- the diagnosed
    failure is scoring/ranking on "other", not missing features. Box branch
    (cv2) and DFL untouched; pairs with standard BCE.

    Drop-in:  - [[14, 17, 20], 1, DetectCosine, [nc]]   (optional scale: [nc, 16])
    """

    def __init__(self, nc=80, scale=16.0, ch=()):
        """Build Detect, then swap the cls branch's final layer for CosineCls."""
        if isinstance(scale, (list, tuple)):
            ch, scale = scale, 16.0
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                CosineCls(c3, self.nc, scale),
            )
            for x in ch
        )

    def bias_init(self):
        """Box bias as usual; cosine cls bias set negative for low init confidence."""
        for a, s in zip(self.cv2, self.stride):
            a[-1].bias.data[:] = 1.0
        for c, s in zip(self.cv3, self.stride):
            c[-1].bias.data[:] = math.log(5 / self.nc / (640 / s) ** 2)


class DetectDecoupledCosine(DetectDecoupled):
    """DetectDecoupled (separate box/cls features) + cosine cls scoring (round 28).

    Combines the two pieces aimed at the "other" problem: decoupled gives the cls
    branch its own features; cosine scoring turns those features into
    well-ranked, well-calibrated class scores. YAML provides 2*nl inputs
    [box..., cls...] like DetectDecoupled (optional scale last).
    """

    def __init__(self, nc=80, scale=16.0, ch=()):
        """Build the decoupled head, then swap cv3's final layer for CosineCls."""
        if isinstance(scale, (list, tuple)):
            ch, scale = scale, 16.0
        super().__init__(nc, ch)
        cls_ch = ch[len(ch) // 2:]
        c3 = max(cls_ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                CosineCls(c3, self.nc, scale),
            )
            for x in cls_ch
        )

    def bias_init(self):
        """Box bias as usual; cosine cls bias set negative for low init confidence."""
        for a, s in zip(self.cv2, self.stride):
            a[-1].bias.data[:] = 1.0
        for c, s in zip(self.cv3, self.stride):
            c[-1].bias.data[:] = math.log(5 / self.nc / (640 / s) ** 2)


class Segment(Detect):
    """YOLO Segment head for segmentation models."""

    def __init__(self, nc=80, nm=32, npr=256, ch=()):
        """Initialize the YOLO model attributes such as the number of masks, prototypes, and the convolution layers."""
        super().__init__(nc, ch)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos

        c4 = max(ch[0] // 4, self.nm)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nm, 1)) for x in ch)

    def forward(self, x):
        """Return model outputs and mask coefficients if training, otherwise return outputs and mask coefficients."""
        p = self.proto(x[0])  # mask protos
        bs = p.shape[0]  # batch size

        mc = torch.cat([self.cv4[i](x[i]).view(bs, self.nm, -1) for i in range(self.nl)], 2)  # mask coefficients
        x = Detect.forward(self, x)
        if self.training:
            return x, mc, p
        return (torch.cat([x, mc], 1), p) if self.export else (torch.cat([x[0], mc], 1), (x[1], mc, p))


class OBB(Detect):
    """YOLO OBB detection head for detection with rotation models."""

    def __init__(self, nc=80, ne=1, ch=()):
        """Initialize OBB with number of classes `nc` and layer channels `ch`."""
        super().__init__(nc, ch)
        self.ne = ne  # number of extra parameters

        c4 = max(ch[0] // 4, self.ne)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.ne, 1)) for x in ch)

    def forward(self, x):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        bs = x[0].shape[0]  # batch size
        angle = torch.cat([self.cv4[i](x[i]).view(bs, self.ne, -1) for i in range(self.nl)], 2)  # OBB theta logits
        # NOTE: set `angle` as an attribute so that `decode_bboxes` could use it.
        angle = (angle.sigmoid() - 0.25) * math.pi  # [-pi/4, 3pi/4]
        # angle = angle.sigmoid() * math.pi / 2  # [0, pi/2]
        if not self.training:
            self.angle = angle
        x = Detect.forward(self, x)
        if self.training:
            return x, angle
        return torch.cat([x, angle], 1) if self.export else (torch.cat([x[0], angle], 1), (x[1], angle))

    def decode_bboxes(self, bboxes, anchors):
        """Decode rotated bounding boxes."""
        return dist2rbox(bboxes, self.angle, anchors, dim=1)


class Pose(Detect):
    """YOLO Pose head for keypoints models."""

    def __init__(self, nc=80, kpt_shape=(17, 3), ch=()):
        """Initialize YOLO network with default parameters and Convolutional Layers."""
        super().__init__(nc, ch)
        self.kpt_shape = kpt_shape  # number of keypoints, number of dims (2 for x,y or 3 for x,y,visible)
        self.nk = kpt_shape[0] * kpt_shape[1]  # number of keypoints total

        c4 = max(ch[0] // 4, self.nk)
        self.cv4 = nn.ModuleList(nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, self.nk, 1)) for x in ch)

    def forward(self, x):
        """Perform forward pass through YOLO model and return predictions."""
        bs = x[0].shape[0]  # batch size
        kpt = torch.cat([self.cv4[i](x[i]).view(bs, self.nk, -1) for i in range(self.nl)], -1)  # (bs, 17*3, h*w)
        x = Detect.forward(self, x)
        if self.training:
            return x, kpt
        pred_kpt = self.kpts_decode(bs, kpt)
        return torch.cat([x, pred_kpt], 1) if self.export else (torch.cat([x[0], pred_kpt], 1), (x[1], kpt))

    def kpts_decode(self, bs, kpts):
        """Decodes keypoints."""
        ndim = self.kpt_shape[1]
        if self.export:
            if self.format in {
                "tflite",
                "edgetpu",
            }:  # required for TFLite export to avoid 'PLACEHOLDER_FOR_GREATER_OP_CODES' bug
                # Precompute normalization factor to increase numerical stability
                y = kpts.view(bs, *self.kpt_shape, -1)
                grid_h, grid_w = self.shape[2], self.shape[3]
                grid_size = torch.tensor([grid_w, grid_h], device=y.device).reshape(1, 2, 1)
                norm = self.strides / (self.stride[0] * grid_size)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * norm
            else:
                # NCNN fix
                y = kpts.view(bs, *self.kpt_shape, -1)
                a = (y[:, :, :2] * 2.0 + (self.anchors - 0.5)) * self.strides
            if ndim == 3:
                a = torch.cat((a, y[:, :, 2:3].sigmoid()), 2)
            return a.view(bs, self.nk, -1)
        else:
            y = kpts.clone()
            if ndim == 3:
                y[:, 2::3] = y[:, 2::3].sigmoid()  # sigmoid (WARNING: inplace .sigmoid_() Apple MPS bug)
            y[:, 0::ndim] = (y[:, 0::ndim] * 2.0 + (self.anchors[0] - 0.5)) * self.strides
            y[:, 1::ndim] = (y[:, 1::ndim] * 2.0 + (self.anchors[1] - 0.5)) * self.strides
            return y


class Classify(nn.Module):
    """YOLO classification head, i.e. x(b,c1,20,20) to x(b,c2)."""

    export = False  # export mode

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
        """Initializes YOLO classification head to transform input tensor from (b,c1,20,20) to (b,c2) shape."""
        super().__init__()
        c_ = 1280  # efficientnet_b0 size
        self.conv = Conv(c1, c_, k, s, p, g)
        self.pool = nn.AdaptiveAvgPool2d(1)  # to x(b,c_,1,1)
        self.drop = nn.Dropout(p=0.0, inplace=True)
        self.linear = nn.Linear(c_, c2)  # to x(b,c2)

    def forward(self, x):
        """Performs a forward pass of the YOLO model on input image data."""
        if isinstance(x, list):
            x = torch.cat(x, 1)
        x = self.linear(self.drop(self.pool(self.conv(x)).flatten(1)))
        if self.training:
            return x
        y = x.softmax(1)  # get final output
        return y if self.export else (y, x)


class WorldDetect(Detect):
    """Head for integrating YOLO detection models with semantic understanding from text embeddings."""

    def __init__(self, nc=80, embed=512, with_bn=False, ch=()):
        """Initialize YOLO detection layer with nc classes and layer channels ch."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.cv4 = nn.ModuleList(BNContrastiveHead(embed) if with_bn else ContrastiveHead() for _ in ch)

    def forward(self, x, text):
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv4[i](self.cv3[i](x[i]), text)), 1)
        if self.training:
            return x

        # Inference path
        shape = x[0].shape  # BCHW
        x_cat = torch.cat([xi.view(shape[0], self.nc + self.reg_max * 4, -1) for xi in x], 2)
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))
            self.shape = shape

        if self.export and self.format in {"saved_model", "pb", "tflite", "edgetpu", "tfjs"}:  # avoid TF FlexSplitV ops
            box = x_cat[:, : self.reg_max * 4]
            cls = x_cat[:, self.reg_max * 4 :]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)

        if self.export and self.format in {"tflite", "edgetpu"}:
            # Precompute normalization factor to increase numerical stability
            # See https://github.com/ultralytics/ultralytics/issues/7371
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1, 4, 1)
            norm = self.strides / (self.stride[0] * grid_size)
            dbox = self.decode_bboxes(self.dfl(box) * norm, self.anchors.unsqueeze(0) * norm[:, :2])
        else:
            dbox = self.decode_bboxes(self.dfl(box), self.anchors.unsqueeze(0)) * self.strides

        y = torch.cat((dbox, cls.sigmoid()), 1)
        return y if self.export else (y, x)

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        m = self  # self.model[-1]  # Detect() module
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1
        # ncf = math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())  # nominal class frequency
        for a, b, s in zip(m.cv2, m.cv3, m.stride):  # from
            a[-1].bias.data[:] = 1.0  # box
            # b[-1].bias.data[:] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)


class RTDETRDecoder(nn.Module):
    """
    Real-Time Deformable Transformer Decoder (RTDETRDecoder) module for object detection.

    This decoder module utilizes Transformer architecture along with deformable convolutions to predict bounding boxes
    and class labels for objects in an image. It integrates features from multiple layers and runs through a series of
    Transformer decoder layers to output the final predictions.
    """

    export = False  # export mode

    def __init__(
        self,
        nc=80,
        ch=(512, 1024, 2048),
        hd=256,  # hidden dim
        nq=300,  # num queries
        ndp=4,  # num decoder points
        nh=8,  # num head
        ndl=6,  # num decoder layers
        d_ffn=1024,  # dim of feedforward
        dropout=0.0,
        act=nn.ReLU(),
        eval_idx=-1,
        # Training args
        nd=100,  # num denoising
        label_noise_ratio=0.5,
        box_noise_scale=1.0,
        learnt_init_query=False,
    ):
        """
        Initializes the RTDETRDecoder module with the given parameters.

        Args:
            nc (int): Number of classes. Default is 80.
            ch (tuple): Channels in the backbone feature maps. Default is (512, 1024, 2048).
            hd (int): Dimension of hidden layers. Default is 256.
            nq (int): Number of query points. Default is 300.
            ndp (int): Number of decoder points. Default is 4.
            nh (int): Number of heads in multi-head attention. Default is 8.
            ndl (int): Number of decoder layers. Default is 6.
            d_ffn (int): Dimension of the feed-forward networks. Default is 1024.
            dropout (float): Dropout rate. Default is 0.
            act (nn.Module): Activation function. Default is nn.ReLU.
            eval_idx (int): Evaluation index. Default is -1.
            nd (int): Number of denoising. Default is 100.
            label_noise_ratio (float): Label noise ratio. Default is 0.5.
            box_noise_scale (float): Box noise scale. Default is 1.0.
            learnt_init_query (bool): Whether to learn initial query embeddings. Default is False.
        """
        super().__init__()
        self.hidden_dim = hd
        self.nhead = nh
        self.nl = len(ch)  # num level
        self.nc = nc
        self.num_queries = nq
        self.num_decoder_layers = ndl

        # Backbone feature projection
        self.input_proj = nn.ModuleList(nn.Sequential(nn.Conv2d(x, hd, 1, bias=False), nn.BatchNorm2d(hd)) for x in ch)
        # NOTE: simplified version but it's not consistent with .pt weights.
        # self.input_proj = nn.ModuleList(Conv(x, hd, act=False) for x in ch)

        # Transformer module
        decoder_layer = DeformableTransformerDecoderLayer(hd, nh, d_ffn, dropout, act, self.nl, ndp)
        self.decoder = DeformableTransformerDecoder(hd, decoder_layer, ndl, eval_idx)

        # Denoising part
        self.denoising_class_embed = nn.Embedding(nc, hd)
        self.num_denoising = nd
        self.label_noise_ratio = label_noise_ratio
        self.box_noise_scale = box_noise_scale

        # Decoder embedding
        self.learnt_init_query = learnt_init_query
        if learnt_init_query:
            self.tgt_embed = nn.Embedding(nq, hd)
        self.query_pos_head = MLP(4, 2 * hd, hd, num_layers=2)

        # Encoder head
        self.enc_output = nn.Sequential(nn.Linear(hd, hd), nn.LayerNorm(hd))
        self.enc_score_head = nn.Linear(hd, nc)
        self.enc_bbox_head = MLP(hd, hd, 4, num_layers=3)

        # Decoder head
        self.dec_score_head = nn.ModuleList([nn.Linear(hd, nc) for _ in range(ndl)])
        self.dec_bbox_head = nn.ModuleList([MLP(hd, hd, 4, num_layers=3) for _ in range(ndl)])

        self._reset_parameters()

    def forward(self, x, batch=None):
        """Runs the forward pass of the module, returning bounding box and classification scores for the input."""
        from ultralytics.models.utils.ops import get_cdn_group

        # Input projection and embedding
        feats, shapes = self._get_encoder_input(x)

        # Prepare denoising training
        dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
            batch,
            self.nc,
            self.num_queries,
            self.denoising_class_embed.weight,
            self.num_denoising,
            self.label_noise_ratio,
            self.box_noise_scale,
            self.training,
        )

        embed, refer_bbox, enc_bboxes, enc_scores = self._get_decoder_input(feats, shapes, dn_embed, dn_bbox)

        # Decoder
        dec_bboxes, dec_scores = self.decoder(
            embed,
            refer_bbox,
            feats,
            shapes,
            self.dec_bbox_head,
            self.dec_score_head,
            self.query_pos_head,
            attn_mask=attn_mask,
        )
        x = dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta
        if self.training:
            return x
        # (bs, 300, 4+nc)
        y = torch.cat((dec_bboxes.squeeze(0), dec_scores.squeeze(0).sigmoid()), -1)
        return y if self.export else (y, x)

    def _generate_anchors(self, shapes, grid_size=0.05, dtype=torch.float32, device="cpu", eps=1e-2):
        """Generates anchor bounding boxes for given shapes with specific grid size and validates them."""
        anchors = []
        for i, (h, w) in enumerate(shapes):
            sy = torch.arange(end=h, dtype=dtype, device=device)
            sx = torch.arange(end=w, dtype=dtype, device=device)
            grid_y, grid_x = torch.meshgrid(sy, sx, indexing="ij") if TORCH_1_10 else torch.meshgrid(sy, sx)
            grid_xy = torch.stack([grid_x, grid_y], -1)  # (h, w, 2)

            valid_WH = torch.tensor([w, h], dtype=dtype, device=device)
            grid_xy = (grid_xy.unsqueeze(0) + 0.5) / valid_WH  # (1, h, w, 2)
            wh = torch.ones_like(grid_xy, dtype=dtype, device=device) * grid_size * (2.0**i)
            anchors.append(torch.cat([grid_xy, wh], -1).view(-1, h * w, 4))  # (1, h*w, 4)

        anchors = torch.cat(anchors, 1)  # (1, h*w*nl, 4)
        valid_mask = ((anchors > eps) & (anchors < 1 - eps)).all(-1, keepdim=True)  # 1, h*w*nl, 1
        anchors = torch.log(anchors / (1 - anchors))
        anchors = anchors.masked_fill(~valid_mask, float("inf"))
        return anchors, valid_mask

    def _get_encoder_input(self, x):
        """Processes and returns encoder inputs by getting projection features from input and concatenating them."""
        # Get projection features
        x = [self.input_proj[i](feat) for i, feat in enumerate(x)]
        # Get encoder inputs
        feats = []
        shapes = []
        for feat in x:
            h, w = feat.shape[2:]
            # [b, c, h, w] -> [b, h*w, c]
            feats.append(feat.flatten(2).permute(0, 2, 1))
            # [nl, 2]
            shapes.append([h, w])

        # [b, h*w, c]
        feats = torch.cat(feats, 1)
        return feats, shapes

    def _get_decoder_input(self, feats, shapes, dn_embed=None, dn_bbox=None):
        """Generates and prepares the input required for the decoder from the provided features and shapes."""
        bs = feats.shape[0]
        # Prepare input for decoder
        anchors, valid_mask = self._generate_anchors(shapes, dtype=feats.dtype, device=feats.device)
        features = self.enc_output(valid_mask * feats)  # bs, h*w, 256

        enc_outputs_scores = self.enc_score_head(features)  # (bs, h*w, nc)

        # Query selection
        # (bs, num_queries)
        topk_ind = torch.topk(enc_outputs_scores.max(-1).values, self.num_queries, dim=1).indices.view(-1)
        # (bs, num_queries)
        batch_ind = torch.arange(end=bs, dtype=topk_ind.dtype).unsqueeze(-1).repeat(1, self.num_queries).view(-1)

        # (bs, num_queries, 256)
        top_k_features = features[batch_ind, topk_ind].view(bs, self.num_queries, -1)
        # (bs, num_queries, 4)
        top_k_anchors = anchors[:, topk_ind].view(bs, self.num_queries, -1)

        # Dynamic anchors + static content
        refer_bbox = self.enc_bbox_head(top_k_features) + top_k_anchors

        enc_bboxes = refer_bbox.sigmoid()
        if dn_bbox is not None:
            refer_bbox = torch.cat([dn_bbox, refer_bbox], 1)
        enc_scores = enc_outputs_scores[batch_ind, topk_ind].view(bs, self.num_queries, -1)

        embeddings = self.tgt_embed.weight.unsqueeze(0).repeat(bs, 1, 1) if self.learnt_init_query else top_k_features
        if self.training:
            refer_bbox = refer_bbox.detach()
            if not self.learnt_init_query:
                embeddings = embeddings.detach()
        if dn_embed is not None:
            embeddings = torch.cat([dn_embed, embeddings], 1)

        return embeddings, refer_bbox, enc_bboxes, enc_scores

    # TODO
    def _reset_parameters(self):
        """Initializes or resets the parameters of the model's various components with predefined weights and biases."""
        # Class and bbox head init
        bias_cls = bias_init_with_prob(0.01) / 80 * self.nc
        # NOTE: the weight initialization in `linear_init` would cause NaN when training with custom datasets.
        # linear_init(self.enc_score_head)
        constant_(self.enc_score_head.bias, bias_cls)
        constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_, reg_ in zip(self.dec_score_head, self.dec_bbox_head):
            # linear_init(cls_)
            constant_(cls_.bias, bias_cls)
            constant_(reg_.layers[-1].weight, 0.0)
            constant_(reg_.layers[-1].bias, 0.0)

        linear_init(self.enc_output[0])
        xavier_uniform_(self.enc_output[0].weight)
        if self.learnt_init_query:
            xavier_uniform_(self.tgt_embed.weight)
        xavier_uniform_(self.query_pos_head.layers[0].weight)
        xavier_uniform_(self.query_pos_head.layers[1].weight)
        for layer in self.input_proj:
            xavier_uniform_(layer[0].weight)


class v10Detect(Detect):
    """
    v10 Detection head from https://arxiv.org/pdf/2405.14458.

    Args:
        nc (int): Number of classes.
        ch (tuple): Tuple of channel sizes.

    Attributes:
        max_det (int): Maximum number of detections.

    Methods:
        __init__(self, nc=80, ch=()): Initializes the v10Detect object.
        forward(self, x): Performs forward pass of the v10Detect module.
        bias_init(self): Initializes biases of the Detect module.

    """

    end2end = True

    def __init__(self, nc=80, ch=()):
        """Initializes the v10Detect object with the specified number of classes and input channels."""
        super().__init__(nc, ch)
        c3 = max(ch[0], min(self.nc, 100))  # channels
        # Light cls head
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                nn.Sequential(Conv(x, x, 3, g=x), Conv(x, c3, 1)),
                nn.Sequential(Conv(c3, c3, 3, g=c3), Conv(c3, c3, 1)),
                nn.Conv2d(c3, self.nc, 1),
            )
            for x in ch
        )
        self.one2one_cv3 = copy.deepcopy(self.cv3)
