# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad, LuggageCBAM, EMA, SimAM, LSKA, DCBAM, DCBAM_MS, ShapeCBAM, DeformableConv2d
from .transformer import TransformerBlock

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "C2fCBAM",
    "C2fEMA",
    "C2fSimAM",
    "C2fLSKA",
    "DySample",
    "ZGGlobalContext",
    "ZGGlobalContext2",
    "ZGGatherContext",
    "C2fDCBAM",
    "C2fShapeCBAM",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCBAM",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "ZGLSKA",
    "ZGGC",
    "ZGSE",
    "ZGMHSA",
    "ZGP2Fuse",
    "ZGStrip",
    "ZGDCN",
    "ZGStar",
    "ZGDSConv",
    "ZGLSKASG",
    "ZGLSKAStripFuse",
    "ZGLSKAMultiDil",
    "ZGLSKAWideFuse",
    "ZGLSKARefine",
    "ZGLSKAExpand",
    "ZGLSKAGCFuse",
    "ZGLSKAWideFuse3",
    "ZGLSKACompactFuse",
    "ZGLSKASelectFuse",
    "ZGSmallDetail",
    "ZGLSKAWideFuseV2",
    "WeightedConcat",
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCBAM(nn.Module):
    """
    Bottleneck with LuggageCBAM attention for enhanced luggage detection.
    
    Applies LuggageCBAM attention after the bottleneck convolutions to help
    the model focus on relevant features for detecting backpacks, bags, and trolleys.
    """

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, reduction=16):
        """
        Initialize BottleneckCBAM.
        
        Args:
            c1: Input channels
            c2: Output channels
            shortcut: Whether to use residual connection
            g: Groups for grouped convolution
            k: Kernel sizes for the two convolutions
            e: Expansion ratio
            reduction: Channel reduction ratio for attention MLP
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.attn = LuggageCBAM(c2, reduction=reduction, residual=False)  # residual handled by bottleneck
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass with attention after convolutions."""
        out = self.attn(self.cv2(self.cv1(x)))
        return x + out if self.add else out


class C2fCBAM(nn.Module):
    """
    C2f module with LuggageCBAM attention for enhanced luggage detection.
    V1 - Original heavy version (kept for compatibility)
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, reduction=16):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            BottleneckCBAM(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0, reduction=reduction) 
            for _ in range(n)
        )
        self.final_attn = LuggageCBAM(c2, reduction=reduction, residual=True)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.final_attn(out)

    def forward_split(self, x):
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        return self.final_attn(out)


class C2fEMA(nn.Module):
    """
    C2f with Efficient Multi-scale Attention (EMA).
    
    Lighter and more effective than C2fCBAM:
    - Standard Bottleneck blocks (no attention inside)
    - Single EMA attention after output conv
    - Better gradient flow, faster training
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
        self.attn = EMA(c2, reduction=8)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))


class C2fSimAM(nn.Module):
    """
    C2f with SimAM (parameter-free attention).
    
    Lightest option - zero additional parameters:
    - Standard Bottleneck blocks
    - SimAM attention (no learnable params)
    - Best for when you want minimal overhead
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
        self.attn = SimAM()

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))


class C2fLSKA(nn.Module):
    """
    C2f with Large Separable Kernel Attention (LSKA).
    
    Uses large receptive fields for capturing luggage at various scales:
    - Standard Bottleneck blocks  
    - LSKA with 7x7 decomposed kernel
    - Good for detecting both small and large luggage
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k_size=7):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
        self.attn = LSKA(c2, k_size=k_size)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))


class C2fDCBAM(nn.Module):
    """
    C2f with Deformable CBAM (D-CBAM) for shape-adaptive luggage detection.
    
    Key innovation for paper:
    - Deformable spatial attention adapts to luggage shapes
    - Trolleys (tall): vertical sampling pattern
    - Bags (wide): horizontal sampling pattern
    - Backpacks (square): uniform sampling pattern
    
    Complements SATAL loss (size-adaptive) with shape-adaptive attention.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, multi_scale=False):
        """
        Args:
            c1: Input channels
            c2: Output channels
            n: Number of Bottleneck blocks
            shortcut: Whether to use residual connections
            g: Groups for grouped convolution
            e: Expansion ratio
            multi_scale: Use DCBAM_MS instead of DCBAM (default: False)
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
        # Shape-adaptive attention
        self.attn = DCBAM_MS(c2) if multi_scale else DCBAM(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))


class C2fShapeCBAM(nn.Module):
    """
    C2f with Shape-Adaptive CBAM (S-CBAM) for luggage detection.
    
    STABLE ALTERNATIVE to C2fDCBAM - no deformable convolutions required.
    
    Key innovation for paper:
    - Shape-specific convolutions (H/V/Square) mixed based on content
    - Trolleys (tall): emphasizes vertical convolution
    - Bags (wide): emphasizes horizontal convolution  
    - Backpacks (square): emphasizes square convolution
    
    Complements SATAL loss (size-adaptive) with shape-adaptive attention.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """
        Args:
            c1: Input channels
            c2: Output channels
            n: Number of Bottleneck blocks
            shortcut: Whether to use residual connections
            g: Groups for grouped convolution
            e: Expansion ratio
        """
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) 
            for _ in range(n)
        )
        # Shape-adaptive attention (no deformable conv - stable)
        self.attn = ShapeCBAM(c2)

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))

    def forward_split(self, x):
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.attn(self.cv2(torch.cat(y, 1)))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        c1 (int): Input channels.
        c2 (): Output channels.
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, c1, c2, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())[:-truncate]
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*layers)
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """Forward pass through the model."""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y

import logging
logger = logging.getLogger(__name__)

USE_FLASH_ATTN = False
try:
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:  # Ampere or newer
        from flash_attn.flash_attn_interface import flash_attn_func
        USE_FLASH_ATTN = True
    else:
        from torch.nn.functional import scaled_dot_product_attention as sdpa
        logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")
except Exception:
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    logger.warning("FlashAttention is not available on this device. Using scaled_dot_product_attention instead.")

class AAttn(nn.Module):
    """
    Area-attention module with the requirement of flash attention.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1.

    Methods:
        forward: Performs a forward process of input tensor and outputs a tensor after the execution of the area attention mechanism.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import AAttn
        >>> model = AAttn(dim=64, num_heads=2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.

    """

    def __init__(self, dim, num_heads, area=1):
        """Initializes the area-attention module, a simple yet efficient attention module for YOLO."""
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qk = Conv(dim, all_head_dim * 2, 1, act=False)
        self.v = Conv(dim, all_head_dim, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)

        self.pe = Conv(all_head_dim, dim, 5, 1, 2, g=dim, act=False)


    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention"""
        B, C, H, W = x.shape
        N = H * W

        qk = self.qk(x).flatten(2).transpose(1, 2)
        v = self.v(x)
        pp = self.pe(v)
        v = v.flatten(2).transpose(1, 2)

        if self.area > 1:
            qk = qk.reshape(B * self.area, N // self.area, C * 2)
            v = v.reshape(B * self.area, N // self.area, C)
            B, N, _ = qk.shape
        q, k = qk.split([C, C], dim=2)

        if x.is_cuda and USE_FLASH_ATTN:
            q = q.view(B, N, self.num_heads, self.head_dim)
            k = k.view(B, N, self.num_heads, self.head_dim)
            v = v.view(B, N, self.num_heads, self.head_dim)

            x = flash_attn_func(
                q.contiguous().half(),
                k.contiguous().half(),
                v.contiguous().half()
            ).to(q.dtype)
        else:
            q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)
            v = v.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)

            attn = (q.transpose(-2, -1) @ k) * (self.head_dim ** -0.5)
            max_attn = attn.max(dim=-1, keepdim=True).values
            exp_attn = torch.exp(attn - max_attn)
            attn = exp_attn / exp_attn.sum(dim=-1, keepdim=True)
            x = (v @ attn.transpose(-2, -1))

            x = x.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return self.proj(x + pp)
    

class ABlock(nn.Module):
    """
    ABlock class implementing a Area-Attention block with effective feature extraction.

    This class encapsulates the functionality for applying multi-head attention with feature map are dividing into areas
    and feed-forward neural network layers.

    Attributes:
        dim (int): Number of hidden channels;
        num_heads (int): Number of heads into which the attention mechanism is divided;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        area (int, optional): Number of areas the feature map is divided.  Defaults to 1.

    Methods:
        forward: Performs a forward pass through the ABlock, applying area-attention and feed-forward layers.

    Examples:
        Create a ABlock and perform a forward pass
        >>> model = ABlock(dim=64, num_heads=2, mlp_ratio=1.2, area=4)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    
    Notes: 
        recommend that dim//num_heads be a multiple of 32 or 64.
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """Initializes the ABlock with area-attention and feed-forward layers for faster feature extraction."""
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Executes a forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        x = x + self.mlp(x)
        return x


class A2C2f(nn.Module):  
    """
    A2C2f module with residual enhanced feature extraction using ABlock blocks with area-attention. Also known as R-ELAN

    This class extends the C2f module by incorporating ABlock blocks for fast attention mechanisms and feature extraction.

    Attributes:
        c1 (int): Number of input channels;
        c2 (int): Number of output channels;
        n (int, optional): Number of 2xABlock modules to stack. Defaults to 1;
        a2 (bool, optional): Whether use area-attention. Defaults to True;
        area (int, optional): Number of areas the feature map is divided. Defaults to 1;
        residual (bool, optional): Whether use the residual (with layer scale). Defaults to False;
        mlp_ratio (float, optional): MLP expansion ratio (or MLP hidden dimension ratio). Defaults to 1.2;
        e (float, optional): Expansion ratio for R-ELAN modules. Defaults to 0.5;
        g (int, optional): Number of groups for grouped convolution. Defaults to 1;
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to True;

    Methods:
        forward: Performs a forward pass through the A2C2f module.

    Examples:
        >>> import torch
        >>> from ultralytics.nn.modules import A2C2f
        >>> model = A2C2f(c1=64, c2=64, n=2, a2=True, area=4, residual=True, e=0.5)
        >>> x = torch.randn(2, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        # num_heads = c_ // 64 if c_ // 64 >= 2 else c_ // 32
        num_heads = c_ // 32

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)  # optional act=FReLU(c2)

        init_values = 0.01  # or smaller
        self.gamma = nn.Parameter(init_values * torch.ones((c2)), requires_grad=True) if a2 and residual else None

        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, num_heads, mlp_ratio, area) for _ in range(2))) if a2 else C3k(c_, c_, 2, shortcut, g) for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        if self.gamma is not None:
            return x + self.gamma.view(1, -1, 1, 1) * self.cv2(torch.cat(y, 1))
        return self.cv2(torch.cat(y, 1))


# =============================================================================
# Zero-Init Gated (ZG) blocks — appended residual branches: y = x + gamma*f(x)
# with gamma initialized to 0 (exact identity at init, full pretrained
# weight transfer when appended after layer 20 in the YAML).
# See runs_noaug_weapon70/gated_blocks.py for design rationale.
# All take (c1, c2, ...) with c2 == c1 (channel-preserving).
# =============================================================================


class ZGLKA(nn.Module):
    """Decomposed Large-Kernel Attention primitive (VAN-style) for ZGLSKA.

    5x5 depthwise -> kxk depthwise dilated(3) -> 1x1 pointwise, used as a
    multiplicative attention map. Effective RF ~ 4 + 3*(k-1) + 1 cells.
    """

    def __init__(self, c, k=7):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 5, 1, 2, groups=c)
        self.dwd = nn.Conv2d(c, c, k, 1, ((k - 1) // 2) * 3, groups=c, dilation=3)
        self.pw = nn.Conv2d(c, c, 1)

    def forward(self, x):
        return self.pw(self.dwd(self.dw(x))) * x


class ZGLSKA(nn.Module):
    """Zero-gated large-kernel context branch. y = x + gamma * f(x), gamma=0.

    f = 1x1 -> SiLU -> ZGLKA(k) -> 1x1. Unlike C2fLSKA this does NOT replace
    a pretrained block — it is appended after it in the YAML.

    YAML args: [c2, k]  e.g. [512, 7]  (c2 is width-scaled by parse_model)
    """

    def __init__(self, c1, c2, k=7):
        super().__init__()
        assert c1 == c2, "ZGLSKA preserves channels (set YAML c2 = input channels)"
        self.pw1 = nn.Conv2d(c1, c1, 1)
        self.act = nn.SiLU()
        self.lka = ZGLKA(c1, k)
        self.pw2 = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        return x + self.gamma * self.pw2(self.lka(self.act(self.pw1(x))))


class ZGGC(nn.Module):
    """Zero-gated Global Context block (GCNet-style) — for P5 / large objects.

    Softmax-pooled global context vector -> bottleneck transform ->
    broadcast-added back, behind a zero gate.

    YAML args: [c2, r]  e.g. [1024, 8]
    """

    def __init__(self, c1, c2, r=8):
        super().__init__()
        assert c1 == c2, "ZGGC preserves channels"
        c_ = max(c1 // r, 16)
        self.attn = nn.Conv2d(c1, 1, 1)
        self.transform = nn.Sequential(
            nn.Conv2d(c1, c_, 1),
            nn.GroupNorm(1, c_),
            nn.SiLU(),
            nn.Conv2d(c_, c1, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        w_ = self.attn(x).view(b, 1, h * w).softmax(dim=-1)  # b,1,hw
        ctx = (x.view(b, c, h * w) @ w_.transpose(1, 2)).view(b, c, 1, 1)
        return x + self.gamma * self.transform(ctx)


class ZGSE(nn.Module):
    """Zero-gated Squeeze-Excitation. Cheapest gated control variant.

    y = x + gamma * (SE(x) * x).

    YAML args: [c2, r]  e.g. [512, 8]
    """

    def __init__(self, c1, c2, r=8):
        super().__init__()
        assert c1 == c2, "ZGSE preserves channels"
        c_ = max(c1 // r, 16)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c_, 1),
            nn.SiLU(),
            nn.Conv2d(c_, c1, 1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        return x + self.gamma * (self.fc(x) * x)


class ZGMHSA(nn.Module):
    """Zero-gated multi-head self-attention — intended for P5 (20x20 tokens).

    DW 3x3 on V as positional encoding (as in PSA blocks).

    YAML args: [c2, num_heads]  e.g. [1024, 4]
    """

    def __init__(self, c1, c2, num_heads=4):
        super().__init__()
        assert c1 == c2, "ZGMHSA preserves channels"
        assert c1 % num_heads == 0
        self.nh = num_heads
        self.scale = (c1 // num_heads) ** -0.5
        self.qkv = nn.Conv2d(c1, c1 * 3, 1)
        self.pe = nn.Conv2d(c1, c1, 3, 1, 1, groups=c1)
        self.proj = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x).reshape(b, 3, self.nh, c // self.nh, h * w)
        q, k, v = qkv.unbind(1)  # each: b, nh, d, hw
        attn = (q.transpose(-2, -1) @ k) * self.scale  # b, nh, hw, hw
        attn = attn.softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).reshape(b, c, h, w)
        out = out + self.pe(v.reshape(b, c, h, w))
        return x + self.gamma * self.proj(out)


class ZGStrip(nn.Module):
    """Zero-gated SEPARABLE strip-kernel attention (1xk + kx1) — for elongated objects.

    Wraps the proven conv.LSKA primitive (separable horizontal+vertical large
    kernel, used in the luggage work) in a zero-init gate:
        y = x + gamma * LSKA_sep(x),  gamma = 0 at init.
    Rationale (weapon_noaug): square dilated kernels (ZGLKA) dilute context
    over background for elongated objects; strips match long_gun/knife
    geometry (median H/W 1.3, long_gun 29% of data). Strip RF k=23 spans
    most of the P4 map directionally at negligible cost (depthwise 1xk+kx1).

    YAML args: [c2, k]  e.g.  [512, 23]
    """

    def __init__(self, c1, c2, k=23):
        super().__init__()
        assert c1 == c2, "ZGStrip preserves channels"
        self.attn = LSKA(c1, k_size=k)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        return x + self.gamma * self.attn(x)


class ZGDCN(nn.Module):
    """Zero-gated Deformable Convolution — ADAPTIVE spatial context.

    y = x + gamma * DCNv2(x), gamma = 0 at init.
    Unlike fixed large kernels (ZGLSKA: same square/strip view everywhere),
    the DCN predicts per-position sampling offsets: the receptive field bends
    along a diagonal long_gun, clusters on a pistol, stays local on
    background. Generalizes the kernel-size/shape search (k7/k11/k15/strip)
    into a learned, per-object view.

    Stability: DeformableConv2d already zero-inits its offsets (starts as a
    plain 3x3), and the zero gate keeps the net identity at init — offsets
    can learn while the gate is still nearly closed.

    YAML args: [c2, k]  e.g.  [512, 3]
    """

    def __init__(self, c1, c2, k=3):
        super().__init__()
        assert c1 == c2, "ZGDCN preserves channels"
        self.dcn = DeformableConv2d(c1, c1, k=k, s=1, p=k // 2)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        return x + self.gamma * self.act(self.bn(self.dcn(x)))


class ZGP2Fuse(nn.Module):
    """Zero-gated P2 -> P3 detail fusion (small-object enrichment).

    Injects high-resolution backbone P2 features (160x160) into the P3 head
    (80x80) as a zero-gated residual: p3 + gamma * refine(down(p2)).
    Gives P3 finer spatial detail for small objects WITHOUT adding a P2
    detection head (which ~374 small train instances cannot support).
    Identity at init -> baseline behavior preserved, pretrained loads fully.

    YAML (two inputs, like Concat):
        - [[14, 2], 1, ZGP2Fuse, []]   # f[0]=P3 head, f[1]=P2 backbone
    parse_model must pass channels: args = [ch[f[0]], ch[f[1]]].
    """

    def __init__(self, c_p3, c_p2):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(c_p2, c_p3, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c_p3),
            nn.SiLU(),
        )
        self.refine = nn.Conv2d(c_p3, c_p3, 3, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(c_p3, 1, 1))

    def forward(self, x):
        p3, p2 = x[0], x[1]
        return p3 + self.gamma * self.refine(self.down(p2))


class ZGStar(nn.Module):
    """Zero-gated STAR block — multiplicative feature mixing (StarNet, 2024).

    y = x + gamma * proj_out(act(proj1(z)) * proj2(z)), gamma = 0 at init,
    where z = BN(DWConv7x7(x)).

    Every ZG block so far (LSKA/GC/SE/MHSA/Strip/DCN) is a variant of spatial
    attention / large-kernel context -- the SKA family. ZGStar uses NO
    spatial-attention map at all: two parallel 1x1 convs project to a wide
    hidden dim and are multiplied element-wise (the "star operation").
    This element-wise product implicitly realizes a high-dimensional
    polynomial feature expansion in a low-dim space (Ma et al., StarNet,
    2024) -- a fundamentally different nonlinearity (multiplicative feature
    interaction) than additive attention. A depthwise 7x7 conv supplies
    cheap spatial context before the star op.

    YAML args: [c2, hidden_mult]  e.g. [512, 4]  (hidden = c1 * hidden_mult)
    """

    def __init__(self, c1, c2, hidden_mult=4):
        super().__init__()
        assert c1 == c2, "ZGStar preserves channels"
        c_hidden = c1 * hidden_mult
        self.dw = nn.Conv2d(c1, c1, 7, 1, 3, groups=c1, bias=False)
        self.bn = nn.BatchNorm2d(c1)
        self.proj1 = nn.Conv2d(c1, c_hidden, 1)
        self.proj2 = nn.Conv2d(c1, c_hidden, 1)
        self.act = nn.SiLU()
        self.proj_out = nn.Conv2d(c_hidden, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        z = self.bn(self.dw(x))
        y = self.act(self.proj1(z)) * self.proj2(z)
        y = self.proj_out(y)
        return x + self.gamma * y


class ZGDSConv(nn.Module):
    """Zero-gated Dynamic Snake Convolution — shape prior for elongated objects.

    y = x + gamma * pw(act(bn(DSConv_x(x) + DSConv_y(x)))), gamma = 0 at init.

    Dynamic Snake Convolution (Qi et al., 2023, originally for tubular
    vessel segmentation) deforms a 1D kernel along a single axis with
    CUMULATIVE per-tap offsets, so the sampling path "snakes" along whatever
    elongated structure is present. weapon_noaug's long_gun/knife classes are
    intrinsically elongated/thin -- this encodes a different adaptivity prior
    than ZGDCN (independent unconstrained 2D offsets per tap, no path
    continuity) or ZGLSKA (fixed kernel shape). Implemented with
    F.grid_sample (pure PyTorch, no torchvision.ops -> avoids the
    deform_conv2d crash seen with ZGDCN).

    Two branches (kernel snaking along x, kernel snaking along y), each:
      1. predict per-tap offsets (1 scalar per tap) from a 3x3 conv,
         zero-initialized so offsets start at 0 (taps sit on the regular
         grid at init);
      2. cumulative-sum offsets outward from the center tap (snake path);
      3. bilinear-sample the input along the deformed 1D path;
      4. depthwise-combine the K sampled taps -> 1 output per channel.

    YAML args: [c2, k]  e.g. [512, 9]
    """

    def __init__(self, c1, c2, k=9):
        super().__init__()
        assert c1 == c2, "ZGDSConv preserves channels"
        assert k % 2 == 1, "k must be odd"
        self.c1 = c1
        self.k = k
        self.offset_x = nn.Conv2d(c1, k, 3, 1, 1)
        self.offset_y = nn.Conv2d(c1, k, 3, 1, 1)
        nn.init.zeros_(self.offset_x.weight)
        nn.init.zeros_(self.offset_x.bias)
        nn.init.zeros_(self.offset_y.weight)
        nn.init.zeros_(self.offset_y.bias)
        self.weight_x = nn.Parameter(torch.randn(c1, k) * 0.02)
        self.weight_y = nn.Parameter(torch.randn(c1, k) * 0.02)
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.SiLU()
        self.pw = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def _snake_sample(self, x, offsets, weight, axis):
        B, C, H, W = x.shape
        K = self.k
        device, dtype = x.device, x.dtype
        off = torch.tanh(offsets.float())  # (B,K,H,W), bounded to (-1,1) taps
        center = K // 2
        cum = torch.zeros_like(off)

        run = torch.zeros(B, H, W, device=device, dtype=off.dtype)
        for i in range(center, K):
            run = run + off[:, i]
            cum[:, i] = run

        run = torch.zeros(B, H, W, device=device, dtype=off.dtype)
        for i in range(center - 1, -1, -1):
            run = run - off[:, i]
            cum[:, i] = run

        ys = torch.linspace(-1, 1, H, device=device, dtype=off.dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=off.dtype)
        base_y, base_x = torch.meshgrid(ys, xs, indexing="ij")  # (H,W)
        step_x = 2.0 / max(W - 1, 1)
        step_y = 2.0 / max(H - 1, 1)

        out = torch.zeros_like(x)
        x32 = x.float()
        for i in range(K):
            tap = i - center
            if axis == "x":
                grid_x = base_x.unsqueeze(0) + tap * step_x + cum[:, i] * step_x
                grid_y = base_y.unsqueeze(0).expand(B, -1, -1)
            else:
                grid_x = base_x.unsqueeze(0).expand(B, -1, -1)
                grid_y = base_y.unsqueeze(0) + tap * step_y + cum[:, i] * step_y
            grid = torch.stack([grid_x, grid_y], dim=-1)  # (B,H,W,2)
            sampled = F.grid_sample(x32, grid, mode="bilinear", padding_mode="border", align_corners=True)
            w = weight[:, i].view(1, C, 1, 1)
            out = out + (sampled * w).to(dtype)
        return out

    def forward(self, x):
        sx = self._snake_sample(x, self.offset_x(x), self.weight_x, "x")
        sy = self._snake_sample(x, self.offset_y(x), self.weight_y, "y")
        y = self.act(self.bn(sx + sy))
        y = self.pw(y)
        return x + self.gamma * y


class ZGLSKASG(nn.Module):
    """Spatially-gated ZGLSKA -- round 9, built on the round-7 dose-response winner.

    y = x + gamma * sigmoid(spatial_gate(x)) * f(x), gamma = 0 at init.
    f = 1x1 -> SiLU -> ZGLKA(k) -> 1x1   (identical branch to ZGLSKA, k=11
    confirmed as the dose-response peak: k7=79.05, k11=79.19, k15=79.03).

    ZGLSKA applies its (per-channel) gate UNIFORMLY across the whole P4
    feature map -- the network learns "how much" large-kernel context to
    mix in per channel, but not "where". ZGLSKASG adds a per-pixel
    sigmoid mask predicted from the input by a small 3x3 conv, so the
    same k=11 branch can be suppressed over background and emphasized
    around object-shaped regions.

    Identity-at-init is preserved exactly the same way as ZGLSKA: the
    per-channel `gamma` is zero-initialized, so y = x regardless of the
    spatial gate's value at step 0. spatial_gate is ALSO zero-initialized
    (weight+bias=0), so sigmoid(.) = 0.5 uniformly at init -- the spatial
    gate starts as a no-op multiplier (0.5, constant) and only begins to
    differentiate spatially once gamma opens and gradients flow into it.

    YAML args: [c2, k]  e.g. [512, 11]
    """

    def __init__(self, c1, c2, k=11):
        super().__init__()
        assert c1 == c2, "ZGLSKASG preserves channels (set YAML c2 = input channels)"
        self.pw1 = nn.Conv2d(c1, c1, 1)
        self.act = nn.SiLU()
        self.lka = ZGLKA(c1, k)
        self.pw2 = nn.Conv2d(c1, c1, 1)
        self.spatial_gate = nn.Conv2d(c1, 1, 3, 1, 1)
        nn.init.zeros_(self.spatial_gate.weight)
        nn.init.zeros_(self.spatial_gate.bias)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        f = self.pw2(self.lka(self.act(self.pw1(x))))
        g = torch.sigmoid(self.spatial_gate(x))  # (B,1,H,W), starts at 0.5 everywhere
        return x + self.gamma * g * f


class ZGLSKAStripFuse(nn.Module):
    """Round 10 — fuse the two best single-branch shapes (k11 square + strip23)
    in ONE gated branch via channel-split, instead of stacking two gates.

    y = x + gamma * pw2( cat[ ZGLKA(k_sq)(z1), LSKA_strip(k_strip)(z2) ] ),
    gamma = 0 at init, z = act(pw1(x)) split in half along channels.

    Round 7's dose-response found k=11 (square, dilated) is the peak
    (79.19%) and strip k=23 (1x23+23x1, for elongated objects) is a close
    second (79.07%) -- two DIFFERENT receptive-field shapes, both near-best.
    Stacking them as two SEPARATE gated branches hurt (k11+GC@P4 = 78.66,
    two competing gammas). This instead routes half the channels through
    each shape inside a SINGLE branch under ONE gamma -- a different failure
    mode than stacking, so the earlier negative result doesn't directly
    predict this one.

    YAML args: [c2, k_sq, k_strip]  e.g. [512, 11, 23]
    """

    def __init__(self, c1, c2, k_sq=11, k_strip=23):
        super().__init__()
        assert c1 == c2, "ZGLSKAStripFuse preserves channels"
        assert c1 % 2 == 0, "ZGLSKAStripFuse requires an even channel count"
        c_half = c1 // 2
        self.pw1 = nn.Conv2d(c1, c1, 1)
        self.act = nn.SiLU()
        self.lka = ZGLKA(c_half, k_sq)
        self.strip = LSKA(c_half, k_size=k_strip)
        self.pw2 = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        z1, z2 = self.act(self.pw1(x)).chunk(2, dim=1)
        y = torch.cat([self.lka(z1), self.strip(z2)], dim=1)
        return x + self.gamma * self.pw2(y)


class ZGLKAMultiDil(nn.Module):
    """Multi-dilation LKA primitive for ZGLSKAMultiDil.

    5x5 depthwise -> TWO parallel dilated depthwise convs at different
    (k, dilation) -> summed -> 1x1 pointwise, used as a multiplicative
    attention map (same usage pattern as ZGLKA, but two receptive-field
    scales fused into one attention map instead of one).
    """

    def __init__(self, c, k_a=7, d_a=2, k_b=11, d_b=3):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 5, 1, 2, groups=c)
        self.dwd_a = nn.Conv2d(c, c, k_a, 1, ((k_a - 1) // 2) * d_a, groups=c, dilation=d_a)
        self.dwd_b = nn.Conv2d(c, c, k_b, 1, ((k_b - 1) // 2) * d_b, groups=c, dilation=d_b)
        self.pw = nn.Conv2d(c, c, 1)

    def forward(self, x):
        z = self.dw(x)
        return self.pw(self.dwd_a(z) + self.dwd_b(z)) * x


class ZGLSKAMultiDil(nn.Module):
    """Round 10 — single-branch multi-scale LKA: k7/dilation2 AND k11/dilation3
    fused into one attention map under one gamma, instead of picking one k.

    y = x + gamma * pw2(ZGLKAMultiDil(act(pw1(x)))), gamma = 0 at init.

    Round 7's dose-response over a single ZGLKA(k, dilation=3): k7=79.05,
    k11=79.19 (peak), k15=79.03 -- a fairly flat curve near the peak,
    suggesting both the k7 (RF~17 cells) and k11 (RF~35 cells) scales carry
    useful signal individually. This fuses both scales (k7/dilation2,
    RF~17, and k11/dilation3, RF~35 -- the dose-response peak) inside ONE
    branch/gate as a single-branch "ensemble" instead of an either/or choice.

    YAML args: [c2, k_a, d_a, k_b, d_b]  e.g. [512, 7, 2, 11, 3]
    """

    def __init__(self, c1, c2, k_a=7, d_a=2, k_b=11, d_b=3):
        super().__init__()
        assert c1 == c2, "ZGLSKAMultiDil preserves channels"
        self.pw1 = nn.Conv2d(c1, c1, 1)
        self.act = nn.SiLU()
        self.lka = ZGLKAMultiDil(c1, k_a, d_a, k_b, d_b)
        self.pw2 = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        return x + self.gamma * self.pw2(self.lka(self.act(self.pw1(x))))


class ZGLSKAWideFuse(nn.Module):
    """Round 11 [idea 2] — fuse k11 (square) + strip23 WITHOUT channel-starvation.

    Round 10's ZGLSKAStripFuse fused these same two shapes by channel-SPLIT
    (each shape only sees c1/2 channels) and was the worst round-10 result
    (78.27, -0.92 vs k11 alone) -- WORSE than round 7's same-scale gate
    stacking (78.66). Diagnosis: halving per-branch channel width starves
    each LKA shape of capacity more than adding a whole second gated branch
    does.

    This fixes that directly: EXPAND first (pw1: c1 -> 2*c1), so each shape
    gets its own FULL c1-width stream (same width either shape would get
    operating alone, as in r6's k11 @ 79.19 or round 7's strip23 @ 79.07),
    concat back to 2*c1, then pw2: 2*c1 -> c1. Single gamma, zero-init.

    y = x + gamma * pw2( cat[ ZGLKA(k_sq)(z1), LSKA_strip(k_strip)(z2) ] ),
    z1, z2 = act(pw1(x)).chunk(2, dim=1), each c1-wide.

    YAML args: [c2, k_sq, k_strip]  e.g. [512, 11, 23]
    """

    def __init__(self, c1, c2, k_sq=11, k_strip=23):
        super().__init__()
        assert c1 == c2, "ZGLSKAWideFuse preserves channels"
        self.pw1 = nn.Conv2d(c1, 2 * c1, 1)
        self.act = nn.SiLU()
        self.lka = ZGLKA(c1, k_sq)
        self.strip = LSKA(c1, k_size=k_strip)
        self.pw2 = nn.Conv2d(2 * c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        z1, z2 = self.act(self.pw1(x)).chunk(2, dim=1)
        y = torch.cat([self.lka(z1), self.strip(z2)], dim=1)
        return x + self.gamma * self.pw2(y)


class ZGLSKARefine(nn.Module):
    """Round 11 [idea 3] — sequential local-refinement after the proven k11 branch.

    Keeps the dose-response winner (ZGLKA k=11, RF~35 cells, 79.19% alone)
    completely intact and unchanged, then adds one cheap depthwise k_refine
    (default 3x3) conv + SiLU AFTER the LKA attention output, before pw2 --
    a local-detail pass on top of k11's global-context attention map, all
    inside the SAME gated residual with ONE gamma (no parallel competition,
    no channel split).

    y = x + gamma * pw2( SiLU(refine(ZGLKA(k)(act(pw1(x))))) ), gamma = 0.

    YAML args: [c2, k, k_refine]  e.g. [512, 11, 3]
    """

    def __init__(self, c1, c2, k=11, k_refine=3):
        super().__init__()
        assert c1 == c2, "ZGLSKARefine preserves channels"
        self.pw1 = nn.Conv2d(c1, c1, 1)
        self.act = nn.SiLU()
        self.lka = ZGLKA(c1, k)
        self.refine = nn.Conv2d(c1, c1, k_refine, 1, k_refine // 2, groups=c1)
        self.refine_act = nn.SiLU()
        self.pw2 = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        f = self.lka(self.act(self.pw1(x)))
        r = self.refine_act(self.refine(f))
        return x + self.gamma * self.pw2(r)


class ZGLSKAExpand(nn.Module):
    """Round 11 [idea 4] — capacity-expanded k11: is k11 RF-limited or capacity-limited?

    Round 7's k=7/11/15 dose-response (79.05/79.19/79.03) was fairly flat
    near the peak -- suggesting receptive field is no longer the bottleneck.
    This keeps k=11 (the peak) but widens the branch: pw1 projects
    c1 -> expand*c1, ZGLKA(k=11) operates on expand*c1 channels (roughly
    `expand`x the depthwise-conv capacity of r6's branch), pw2 projects back
    expand*c1 -> c1. Single gamma, zero-init, same kernel as the proven
    winner -- only the channel width changes.

    y = x + gamma * pw2( ZGLKA(k)(act(pw1(x))) ),  pw1: c1->e*c1, pw2: e*c1->c1.

    YAML args: [c2, k, expand]  e.g. [512, 11, 2]
    """

    def __init__(self, c1, c2, k=11, expand=2):
        super().__init__()
        assert c1 == c2, "ZGLSKAExpand preserves channels"
        c_wide = c1 * expand
        self.pw1 = nn.Conv2d(c1, c_wide, 1)
        self.act = nn.SiLU()
        self.lka = ZGLKA(c_wide, k)
        self.pw2 = nn.Conv2d(c_wide, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        return x + self.gamma * self.pw2(self.lka(self.act(self.pw1(x))))


class ZGLSKAGCFuse(nn.Module):
    """Round 14 — fuse k_sq LKA (local, proven) with GCNet-style global
    context, full-width streams (WideFuse structure), one gamma.

    ZGLSKAWideFuse's two branches (k=11 LKA + strip-23 LSKA) are BOTH
    large-local-RF shapes -- never tested against a branch that's
    QUALITATIVELY different (globally-pooled context). Round 13's
    DetectLKACls (local k=11, isolated to the cls branch only) consistently
    HURT both backbones it was tried on (-0.38, -0.86 vs their unmodified
    backbones) -- but that tested "local RF, cls-only injection". This tests
    a different combination: "global context, SHARED feature" (affects both
    box and cls, like WideFuse's proven structure) instead of cls-only.

    y = x + gamma * pw2( cat[ ZGLKA(k)(z1), z2 + gc_transform(ctx(z2)) ] ),
    z1, z2 = act(pw1(x)).chunk(2, dim=1), each c1-wide. ctx(z2) is a
    softmax-attention-pooled global context vector (GCNet-style).

    YAML args: [c2, k, r]  e.g. [512, 11, 8]
    """

    def __init__(self, c1, c2, k=11, r=8):
        super().__init__()
        assert c1 == c2, "ZGLSKAGCFuse preserves channels"
        self.pw1 = nn.Conv2d(c1, 2 * c1, 1)
        self.act = nn.SiLU()
        self.lka = ZGLKA(c1, k)
        c_ = max(c1 // r, 16)
        self.gc_attn = nn.Conv2d(c1, 1, 1)
        self.gc_transform = nn.Sequential(
            nn.Conv2d(c1, c_, 1),
            nn.GroupNorm(1, c_),
            nn.SiLU(),
            nn.Conv2d(c_, c1, 1),
        )
        self.pw2 = nn.Conv2d(2 * c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def _gc(self, z):
        b, c, h, w = z.shape
        w_ = self.gc_attn(z).view(b, 1, h * w).softmax(dim=-1)  # b,1,hw
        ctx = (z.view(b, c, h * w) @ w_.transpose(1, 2)).view(b, c, 1, 1)  # b,c,1,1
        return z + self.gc_transform(ctx)

    def forward(self, x):
        z1, z2 = self.act(self.pw1(x)).chunk(2, dim=1)
        y = torch.cat([self.lka(z1), self._gc(z2)], dim=1)
        return x + self.gamma * self.pw2(y)


class ZGLSKAWideFuse3(nn.Module):
    """Round 15 — WideFuse + a NEW small-receptive-field detail branch, one gamma.

    Cross-round finding (17 variants, rounds 6-14 + arch_zg): baseline's
    mAP50_small=61.79% and "other"-class small AP50=38.57% have never been
    approached by ANY tested architecture. The closest is 59.64% mAP50_small
    (a weak overall performer, 78.66%). r11_widefuse_70 (best overall, 79.40%)
    has mAP50_small=56.65% and "other"-small AP50=23.59% -- a ~15pp relative
    drop on "other"-small vs baseline. EVERY P4/head-focused module tried
    (LKA, strip, GC, star, multi-dil, P2/P3/P4 fusions, cls-branch context)
    reproduces this same trade-off regardless of where it sits.

    Diagnosis: ZGLSKAWideFuse's two branches (k=11 square ZGLKA + strip-23
    LSKA) are BOTH large-receptive-field operators -- neither preserves fine,
    small-scale detail at P4-BU (layer 17), which is exactly the capacity
    "other"-small needs.

    Fix: add a THIRD, genuinely small-RF branch (k_small=3 depthwise conv,
    dilation=1, GroupNorm+SiLU -- a pure fine-detail pass) in parallel with
    WideFuse's two proven branches, all under ONE gamma. pw1 expands
    c1 -> 3*c1 so each branch gets its own full c1-width stream (no channel
    starvation, same fix WideFuse itself applied to StripFuse's 2-way split).

    This is a STRICT GENERALIZATION of ZGLSKAWideFuse: at init gamma=0
    (exact identity, same as WideFuse and the same as r11's checkpoint), and
    during training the small-RF branch's contribution can shrink toward zero
    if unhelpful, collapsing back to ~WideFuse behavior. Downside risk is
    "ties r11_widefuse_70 (79.40)", not "regresses below it".

    y = x + gamma * pw2( cat[ ZGLKA(k_sq)(z1), LSKA(k_strip)(z2), small(z3) ] ),
    z1, z2, z3 = act(pw1(x)).chunk(3, dim=1), each c1-wide.
    pw1: c1 -> 3*c1, pw2: 3*c1 -> c1.

    YAML args: [c2, k_sq, k_strip, k_small]  e.g. [512, 11, 23, 3]
    """

    def __init__(self, c1, c2, k_sq=11, k_strip=23, k_small=3):
        super().__init__()
        assert c1 == c2, "ZGLSKAWideFuse3 preserves channels"
        self.pw1 = nn.Conv2d(c1, 3 * c1, 1)
        self.act = nn.SiLU()
        self.lka = ZGLKA(c1, k_sq)
        self.strip = LSKA(c1, k_size=k_strip)
        self.small = nn.Sequential(
            nn.Conv2d(c1, c1, k_small, 1, k_small // 2, groups=c1),
            nn.GroupNorm(1, c1),
            nn.SiLU(),
        )
        self.pw2 = nn.Conv2d(3 * c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        z1, z2, z3 = self.act(self.pw1(x)).chunk(3, dim=1)
        y = torch.cat([self.lka(z1), self.strip(z2), self.small(z3)], dim=1)
        return x + self.gamma * self.pw2(y)


class ZGLSKACompactFuse(nn.Module):
    """Round 16 — WideFuse shape, but swap the elongated strip-23 branch for
    a COMPACT multi-scale SMALL-kernel branch (k3/dilation1 + k5/dilation2).

    Cross-round finding (rounds 6-15): every architecture that fused k=11
    ZGLKA with another LARGE-receptive-field shape (strip-23 LSKA in
    WideFuse, 79.40; GCNet global context in GCFuse, 78.23) traded away
    small-object / "other"-class AP50 for overall mAP50. ZGLSKAWideFuse3
    (round 15) tested ADDING a third small-RF branch alongside both large
    branches -- a strict superset. This module tests the more aggressive
    alternative: directly REPLACING the second large-RF branch (strip-23)
    with a small-RF one, keeping the proven k11 ZGLKA branch and the
    WideFuse 2-branch/expand-then-fuse shape (pw1: c1->2*c1, pw2: 2*c1->c1,
    one gamma) unchanged.

    The "strip-23" role is filled by a compact multi-scale fusion of two
    SMALL dilated depthwise convs (k=3/dilation=1, RF=3, and k=5/dilation=2,
    RF=9) summed, then GroupNorm+SiLU -- cheap, local, multi-scale fine
    detail, in the same spirit as ZGLKAMultiDil (round 10) but at small
    kernel sizes instead of k7/k11.

    y = x + gamma * pw2( cat[ ZGLKA(k_sq)(z1), SiLU(GN(dwA(z2)+dwB(z2))) ] ),
    z1, z2 = act(pw1(x)).chunk(2, dim=1), each c1-wide.
    pw1: c1 -> 2*c1, pw2: 2*c1 -> c1.

    gamma=0 at init -> exact identity, append-only Detect-remap loader
    applies as usual. Unlike WideFuse3 (additive superset of WideFuse),
    this is a DIFFERENT 2-branch architecture at the same parameter budget
    as WideFuse (~same shape, smaller kernels) -- higher risk/reward: if
    "other"-small recovers without losing overall mAP50, it suggests the
    strip-23 branch itself (not just "a second large-RF branch") was the
    binding constraint.

    YAML args: [c2, k_sq, k_a, d_a, k_b, d_b]  e.g. [512, 11, 3, 1, 5, 2]
    """

    def __init__(self, c1, c2, k_sq=11, k_a=3, d_a=1, k_b=5, d_b=2):
        super().__init__()
        assert c1 == c2, "ZGLSKACompactFuse preserves channels"
        self.pw1 = nn.Conv2d(c1, 2 * c1, 1)
        self.act = nn.SiLU()
        self.lka = ZGLKA(c1, k_sq)
        self.branch_a = nn.Conv2d(c1, c1, k_a, 1, ((k_a - 1) // 2) * d_a, groups=c1, dilation=d_a)
        self.branch_b = nn.Conv2d(c1, c1, k_b, 1, ((k_b - 1) // 2) * d_b, groups=c1, dilation=d_b)
        self.compact_norm = nn.GroupNorm(1, c1)
        self.compact_act = nn.SiLU()
        self.pw2 = nn.Conv2d(2 * c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        z1, z2 = self.act(self.pw1(x)).chunk(2, dim=1)
        compact = self.compact_act(self.compact_norm(self.branch_a(z2) + self.branch_b(z2)))
        y = torch.cat([self.lka(z1), compact], dim=1)
        return x + self.gamma * self.pw2(y)


class ZGLSKASelectFuse(nn.Module):
    """Round 17 — spatially content-ADAPTIVE receptive-field routing @ P4-BU.

    Cross-round finding (rounds 6-16): every gated fusion (WideFuse k11+strip23
    = 79.40, GCFuse = 78.23, WideFuse3 additive k3, CompactFuse k3/k5) combines
    its branches with a SINGLE GLOBAL mixing rule -- pw2 over a fixed concat, or
    one per-channel gamma. That weighting is identical at every spatial location,
    so the small-RF branch fires on large objects (noise) and the large-RF
    branch fires on small objects (smoothing). The network is forced into ONE
    global compromise, and the compromise that maximises overall mAP is exactly
    the one that sacrifices small-"other" detail (AP50_small 23.59 vs baseline
    38.57). Searching WHICH branches to fuse cannot escape this; the binding
    constraint is that the fuse is STATIC.

    Innovation: make the branch selection PER-PIXEL and content-dependent. The
    same three branches as WideFuse3 (square ZGLKA k_sq, strip LSKA k_strip,
    small depthwise k_small) are combined not by concat+projection but by a
    lightweight spatial router -> per-location softmax over the 3 branches. The
    receptive field thus adapts to local object scale: small objects route to
    the k_small detail branch (preserving fine structure), large objects route
    to k_sq/k_strip context -- simultaneously, in different regions of the SAME
    P4 map. This is the one degree of freedom every prior fusion lacked.

    Identity / transfer: gamma=0 at init -> exact identity at epoch 0 (full
    Detect-remap pretrained transfer, append-only, like the rest of the family).
    The router weight is zero-init and its bias is warm-started to favour the
    square-LKA branch, so as gamma grows the early behaviour approximates
    r11_widefuse/r6 rather than a random mix.

    Controlled ablation vs ZGLSKAWideFuse3: IDENTICAL three branches and the
    same YAML args -- the ONLY difference is static concat (WideFuse3) vs
    spatial-softmax routing (this) -- isolating "spatially-adaptive receptive
    field" as the mechanism. The learned router weights are directly
    visualisable as a per-location scale map.

    y = x + gamma * pw2( sum_b w_b(x) * branch_b(z_b) ),
    z1,z2,z3 = act(pw1(x)).chunk(3); w = softmax(router(x), dim=branch), spatial.
    pw1: c1 -> 3*c1 (full-width per branch); pw2: c1 -> c1 (post weighted-sum).

    YAML args: [c2, k_sq, k_strip, k_small]  e.g. [512, 11, 23, 3]
    """

    def __init__(self, c1, c2, k_sq=11, k_strip=23, k_small=3):
        super().__init__()
        assert c1 == c2, "ZGLSKASelectFuse preserves channels"
        self.pw1 = nn.Conv2d(c1, 3 * c1, 1)
        self.act = nn.SiLU()
        self.lka = ZGLKA(c1, k_sq)
        self.strip = LSKA(c1, k_size=k_strip)
        self.small = nn.Sequential(
            nn.Conv2d(c1, c1, k_small, 1, k_small // 2, groups=c1),
            nn.GroupNorm(1, c1),
            nn.SiLU(),
        )
        self.router = nn.Conv2d(c1, 3, 1)  # per-location logits over the 3 branches
        self.pw2 = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))
        # warm-start: uniform-ish router that favours the square-LKA branch (idx 0),
        # so once gamma grows the early behaviour approximates r6/r11_widefuse.
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)
        self.router.bias.data[0] = 2.0

    def forward(self, x):
        z1, z2, z3 = self.act(self.pw1(x)).chunk(3, dim=1)
        feats = torch.stack([self.lka(z1), self.strip(z2), self.small(z3)], dim=1)  # B,3,C,H,W
        w = self.router(x).softmax(dim=1).unsqueeze(2)  # B,3,1,H,W (per-location)
        fused = (w * feats).sum(dim=1)  # B,C,H,W
        return x + self.gamma * self.pw2(fused)


class ZGGlobalContext(nn.Module):
    """Global-context block (GCNet / non-local-lite) -- AXIS: global context modeling.

    Per-location features lack whole-image context, which is exactly what the
    heterogeneous "other" class (defined by what it is NOT) needs to be
    disambiguated. This computes a single global descriptor (global average pool),
    transforms it through a small MLP, and broadcasts it back additively into every
    spatial location. Channel-preserving; gamma=0 at init -> exact identity at epoch
    0 (clean pretrained transfer, append-only like the rest of the family).

    YAML: drop-in single-input, e.g.  - [21, 1, ZGGlobalContext, [512]]
          (the channel arg is nominal and width-scaled by parse_model).
    """

    def __init__(self, c1, c2, reduction=8):
        super().__init__()
        assert c1 == c2, "ZGGlobalContext preserves channels"
        hidden = max(8, c1 // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(c1, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, c1, 1)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        ctx = x.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1) global descriptor
        ctx = self.fc(ctx)
        return x + self.gamma * ctx              # gated additive broadcast


class ZGGatherContext(nn.Module):
    """Global CROSS-SCALE context injected into the finest level -- AXIS: gather-
    distribute neck (Gold-YOLO-style global fusion vs the local PAN).

    The standard PAN fuses only adjacent scales (P3 sees P4). Small objects and the
    context-defined "other" class both fail for lack of GLOBAL multi-scale context.
    This gathers a global descriptor from P3, P4 AND P5 (avg-pool each), fuses them,
    and broadcasts the result back into the P3 (small-object) level as a gated
    additive context -- so every P3 location sees whole-scene, all-scale information.

    Multi-input: YAML 'from' = [P3, P4, P5]; output = enhanced P3 (P3 channels).
    gamma=0 at init -> identity. Inference cost is negligible (pooled 1x1 path).

    YAML: - [[14, 17, 20], 1, ZGGatherContext, []]
    """

    def __init__(self, chs):
        super().__init__()
        c_p3, c_p4, c_p5 = chs
        self.proj = nn.Sequential(
            nn.Conv2d(c_p3 + c_p4 + c_p5, c_p3, 1), nn.SiLU(),
            nn.Conv2d(c_p3, c_p3, 1),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        p3, p4, p5 = x
        g = torch.cat([t.mean(dim=(2, 3), keepdim=True) for t in (p3, p4, p5)], dim=1)
        g = self.proj(g)             # (B, c_p3, 1, 1)
        return p3 + self.gamma * g   # broadcast global cross-scale context into P3


class ZGGlobalContext2(nn.Module):
    """globalctx + max-pool branch (avg+max global descriptor) -- the last-arch
    refinement of the round-38 winner.

    ZGGlobalContext built its global descriptor from AVERAGE pool only (whole-scene
    context). This adds a MAX-pool branch: avg captures context, max captures the
    single most SALIENT activation -- the cue small/rare weapon instances spike on
    but that averages away in avg-pool (the CBAM/BAM channel-attention insight).
    Both descriptors are concatenated -> MLP -> gated additive broadcast.

    Stays gentle/gated/identity-init (gamma=0 -> identity at epoch 0), the property
    that made globalctx generalize and the aggressive variants (gather, wfv2_p3)
    fail. It ENRICHES the winning module rather than stacking a second one (which
    is what sank r39). Channel-preserving, ~0 inference cost.

    YAML: drop-in single-input, e.g.  - [21, 1, ZGGlobalContext2, [512]]
    """

    def __init__(self, c1, c2, reduction=8):
        super().__init__()
        assert c1 == c2, "ZGGlobalContext2 preserves channels"
        hidden = max(8, c1 // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(2 * c1, hidden, 1), nn.SiLU(), nn.Conv2d(hidden, c1, 1)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        avg = x.mean(dim=(2, 3), keepdim=True)           # scene context
        mx = x.amax(dim=(2, 3), keepdim=True)            # most salient activation
        ctx = self.fc(torch.cat([avg, mx], dim=1))       # (B, C, 1, 1)
        return x + self.gamma * ctx                      # gated additive broadcast


class DySample(nn.Module):
    """Dynamic content-aware upsampler (DySample, ICCV 2023), 'lp' style.

    Drop-in replacement for nn.Upsample(scale_factor=2) in the FPN top-down path.
    Instead of fixed nearest/bilinear interpolation, it predicts per-location
    sampling offsets and gathers via grid_sample -> recovers fine spatial detail
    when upsampling toward the P3 (small-object) level, where nearest-neighbour
    upsampling blurs exactly the detail small objects need. Channel-preserving;
    the offset conv is near-zero-init so it starts ~ bilinear (safe transfer).

    YAML: drop-in for nn.Upsample ->  - [-1, 1, DySample, [2]]   (scale=2)
    """

    def __init__(self, c1, scale=2, groups=4):
        super().__init__()
        assert c1 % groups == 0, "DySample: channels must be divisible by groups"
        self.scale = scale
        self.groups = groups
        self.offset = nn.Conv2d(c1, 2 * groups * scale * scale, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(
            1, self.groups, 1, 1).reshape(1, -1, 1, 1)

    def forward(self, x):
        offset = self.offset(x) * 0.25 + self.init_pos
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H, device=x.device) + 0.5
        coords_w = torch.arange(W, device=x.device) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])).transpose(
            1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        xg = x.reshape(B * self.groups, -1, H, W)
        return F.grid_sample(xg, coords, mode="bilinear", align_corners=False,
                             padding_mode="border").view(B, -1, self.scale * H, self.scale * W)


class ZGLSKAWideFuseV2(nn.Module):
    """Round 31 -- The Hybrid Branch WideFuse.

    This is the direct, surgical fix for r21's small-object precision problem.
    It keeps the proven two-branch, expand-then-fuse structure of ZGLSKAWideFuse,
    but upgrades the second branch to be a HYBRID of large and small RF operators.

    - Branch 1 (Unchanged): The proven square k=11 LKA for general context.
    - Branch 2 (Hybrid): This branch now has two parallel sub-paths:
        - Large-RF Path: The original strip-23 LKA for elongated objects.
        - Small-RF Path: The ZGSmallDetail logic (k=3 + k=5) for fine detail.
      The outputs of these two sub-paths are ADDED together before returning to
      the main fusion point. This creates a single, powerful Hybrid Branch that
      is effective at all scales, preventing the large-RF operators from
      destroying small-object features at the source.

    y = x + gamma * pw2( cat[ LKA(z1), (strip(z2) + small(z2)) ] ),
    z1, z2 = act(pw1(x)).chunk(2, dim=1), each c1-wide.
    pw1: c1 -> 2*c1, pw2: 2*c1 -> c1. Zero-init gamma as always.

    YAML args: [c2, k_sq, k_strip, k_fine, k_mid]
               e.g. [512, 11, 23, 3, 5]
    """

    def __init__(self, c1, c2, k_sq=11, k_strip=23, k_fine=3, k_mid=5):
        super().__init__()
        assert c1 == c2, "ZGLSKAWideFuseV2 preserves channels"
        self.pw1 = nn.Conv2d(c1, 2 * c1, 1)
        self.act = nn.SiLU()

        # Branch 1: Square LKA (unchanged)
        self.lka = ZGLKA(c1, k_sq)

        # Branch 2: Hybrid (Large RF + Small RF)
        self.strip = LSKA(c1, k_size=k_strip)
        self.dw_fine = nn.Conv2d(c1, c1, k_fine, 1, k_fine // 2, groups=c1)
        self.dw_mid = nn.Conv2d(c1, c1, k_mid, 1, k_mid // 2, groups=c1)
        self.small_norm = nn.GroupNorm(1, c1)
        self.small_act = nn.SiLU()
        
        self.pw2 = nn.Conv2d(2 * c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        z1, z2 = self.act(self.pw1(x)).chunk(2, dim=1)
        
        # Branch 1 output
        y1 = self.lka(z1)

        # Branch 2 (Hybrid) output
        y2_large_rf = self.strip(z2)
        y2_small_rf = self.small_act(self.small_norm(self.dw_fine(z2) + self.dw_mid(z2)))
        y2_hybrid = y2_large_rf + y2_small_rf

        # Fuse and gate
        y = torch.cat([y1, y2_hybrid], dim=1)
        return x + self.gamma * self.pw2(y)


class ZGSmallDetail(nn.Module):
    """Round 30 — zero-gated small-kernel detail guard for P3 features.

    Cross-round finding (rounds 6-29, 90 experiments): r21_widefuse_aux_w50
    (ZGLSKAWideFuse @ P4 + DetectAux, aux_weight=0.5) is the best overall model
    (mAP50=79.57, mAP50-95=50.33) but its small-object mAP50 drops 3.8pp vs
    baseline (57.95 vs 61.79) and "other"-class AP50_small collapses by 12pp
    (26.49 vs 38.57). The model's RECALL on small objects is actually the best
    of all runs (AR50_small=88.69) — it finds them but misclassifies them.

    Root cause: ZGLSKAWideFuse's two branches (k=11 square LKA + strip-23 LSKA)
    are BOTH large-RF operators at P4. During training, gradients from these
    large-RF branches shift shared backbone representations toward medium/large
    object features, degrading the fine-grained P3 features that small objects
    depend on. The P3 head (layer 14) passes through UNTOUCHED to Detect, but
    its upstream backbone features are polluted.

    Fix: a complementary SMALL-kernel gated block placed AFTER layer 14 (P3 head)
    and BEFORE Detect. Two parallel depthwise convolutions (k=3 + k=5, dilation=1)
    capture fine local detail at two micro-scales, summed and projected — NO
    large-RF operator. This counterbalances the widefuse's large-RF bias by
    reinforcing precisely the fine-grained, small-scale features that get
    degraded.

    r16_CompactFuse (which replaced strip-23 with small k3+k5 kernels at P4)
    recovered "other" AP50_small from 23.59 -> 32.87 (+9.3pp), proving small-
    kernel operators DO recover fine detail. But CompactFuse REPLACED the strip-23
    at P4, losing overall mAP50 (79.40 -> 78.25). This approach KEEPS widefuse
    intact at P4 and adds the detail guard SEPARATELY at P3.

    Identity / transfer: gamma=0 at init -> exact identity at epoch 0, append-only
    (standard Detect-remap pretrained loader). The block can only HELP P3 features;
    if unhelpful, gamma stays near zero and the module collapses to a no-op.

    y = x + gamma * pw2( act2( GN( dw3(z) + dw5(z) ) ) ), z = act(pw1(x)).
    pw1: c1 -> c1, pw2: c1 -> c1.

    YAML args: [c2, k_fine, k_mid]  e.g. [256, 3, 5]
    YAML usage (after widefuse at P4):
      - [14, 1, ZGSmallDetail, [256, 3, 5]]        # P3 detail guard
      - [[<p3_guard>, <widefuse>, 20], 1, Detect/DetectAux, [nc, ...]]
    """

    def __init__(self, c1, c2, k_fine=3, k_mid=5):
        super().__init__()
        assert c1 == c2, "ZGSmallDetail preserves channels"
        self.pw1 = nn.Conv2d(c1, c1, 1)
        self.act = nn.SiLU()
        self.dw_fine = nn.Conv2d(c1, c1, k_fine, 1, k_fine // 2, groups=c1)
        self.dw_mid = nn.Conv2d(c1, c1, k_mid, 1, k_mid // 2, groups=c1)
        self.norm = nn.GroupNorm(1, c1)
        self.act2 = nn.SiLU()
        self.pw2 = nn.Conv2d(c1, c1, 1)
        self.gamma = nn.Parameter(torch.zeros(c1, 1, 1))

    def forward(self, x):
        z = self.act(self.pw1(x))
        detail = self.act2(self.norm(self.dw_fine(z) + self.dw_mid(z)))
        return x + self.gamma * self.pw2(detail)


class WeightedConcat(nn.Module):
    """Round 19 -- BiFPN-style learnable weighted fusion (drop-in for Concat).

    The stock YOLOv12 neck fuses scales with hard Concat -- every input branch
    contributes equally and unconditionally. EfficientDet's BiFPN showed that
    making the fusion weights LEARNABLE (per-branch, non-negative) lets the
    network down-weight uninformative scale inputs and emphasise the ones that
    matter, yielding more discriminative fused features without growing the
    backbone. This is the one neck-level lever untouched by the rounds 1-18
    single-module search (which only inserted blocks into a fixed-topology PAN).

    Each input branch i is scaled by a learnable non-negative weight w_i before
    concatenation:  out = cat([relu(w_i) * x_i], dim).
    At init all w_i = 1 -> relu(w_i) = 1 -> EXACTLY standard Concat, so the
    pretrained neck transfers unchanged and the model only *learns* to reweight
    the branches during training (identity-at-init, same discipline as the ZG
    family). Pairs with the BiFPN extra cross-scale edge (3-way fusion at P4)
    in the round-19 YAMLs.

    YAML args: [dimension, n_inputs]  e.g. [1, 2] or [1, 3]
    """

    def __init__(self, dimension=1, n=2):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(n))

    def forward(self, x):
        w = torch.relu(self.w)
        return torch.cat([w[i] * x[i] for i in range(len(x))], self.d)
