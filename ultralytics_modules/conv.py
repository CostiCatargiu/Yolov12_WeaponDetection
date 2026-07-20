# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "ChannelSE",
    "EnhancedChannelAttention",
    "EnhancedSpatialAttention",
    "LuggageCBAM",
    "LuggageCBAMv2",
    "ScaleAdaptiveSpatialAttention",
    "CoordinateAttention",
    "EMA",
    "SimAM", 
    "LSKA",
    "DeformableConv2d",
    "AdaptiveShapeConv",
    "DeformableSpatialAttention",
    "ShapeSpatialAttention",
    "DCBAM",
    "DCBAM_MS",
    "ShapeCBAM",
    "Concat",
    "RepConv",
    "Index",
    "SmallObjectRefinement",
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes Ghost Convolution module with primary and cheap operations for efficient feature learning."""
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class ChannelSE(nn.Module):
    """
    Squeeze-and-Excitation (SE) Channel Attention module.
    
    From SENet (Hu et al., CVPR 2018) - Won ImageNet 2017.
    
    Learns which feature channels are important and amplifies them while
    suppressing unimportant ones. Helps with bag vs backpack confusion by
    amplifying discriminative channels (straps, handles, wheels) and
    suppressing generic ones (textures, edges).
    
    Key properties:
    - Proven: Used in EfficientNet, MobileNetV3, RegNet
    - Minimal risk: Worst case learns ~0.5 weights = identity
    - Tiny overhead: ~82K params for 256+512 channels (0.3% of model)
    - Complementary: A2C2f does spatial attention, ChannelSE does channel attention
    """

    def __init__(self, c1: int, reduction: int = 8) -> None:
        """
        Initialize ChannelSE module.
        
        Args:
            c1: Number of input channels
            reduction: Channel reduction ratio (default 8, so 256->32)
        """
        super().__init__()
        c_mid = max(c1 // reduction, 16)  # At least 16 channels in bottleneck
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),       # (B,C,H,W) -> (B,C,1,1) - global context
            nn.Conv2d(c1, c_mid, 1),       # (B,C,1,1) -> (B,C/r,1,1) - compress
            nn.ReLU(inplace=True),         # Non-linearity
            nn.Conv2d(c_mid, c1, 1),       # (B,C/r,1,1) -> (B,C,1,1) - expand
            nn.Sigmoid()                   # Scale to 0-1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention: amplify important channels, suppress others."""
        return x * self.attn(x)


class EnhancedChannelAttention(nn.Module):
    """
    Enhanced Channel Attention module for luggage detection.
    
    Uses both average and max pooling (per original CBAM paper) with MLP reduction
    for better channel-wise feature recalibration. Optimized for detecting 
    small/medium objects like backpacks, bags, and trolleys.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """
        Initialize Enhanced Channel Attention.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for MLP bottleneck (default: 16)
        """
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP with bottleneck
        reduced_channels = max(channels // reduction, 8)  # Minimum 8 channels
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, 1, bias=False)
        )
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply enhanced channel attention using both avg and max pooling."""
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return x * self.act(avg_out + max_out)


class EnhancedSpatialAttention(nn.Module):
    """
    Enhanced Spatial Attention module for luggage detection.
    
    Uses multi-scale spatial attention with both 3x3 and 7x7 kernels
    to capture both fine-grained details (small objects) and larger context.
    """

    def __init__(self):
        """Initialize Enhanced Spatial Attention with multi-scale kernels."""
        super().__init__()
        # Multi-scale spatial attention: 3x3 for details, 7x7 for context
        self.conv_small = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.conv_large = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.act = nn.Sigmoid()
        
        # Learnable fusion weight
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply multi-scale spatial attention."""
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        
        # Multi-scale attention maps
        attn_small = self.conv_small(spatial_features)
        attn_large = self.conv_large(spatial_features)
        
        # Learnable fusion of scales
        w = torch.sigmoid(self.fusion_weight)
        fused_attn = w * attn_small + (1 - w) * attn_large
        
        return x * self.act(fused_attn)


class LuggageCBAM(nn.Module):
    """
    Enhanced CBAM optimized for luggage detection (backpack, bag, trolley).
    
    V1 - Original version with class-aware bias (kept for compatibility)
    """

    def __init__(self, c1: int, reduction: int = 16, residual: bool = True):
        super().__init__()
        self.channel_attention = EnhancedChannelAttention(c1, reduction)
        self.spatial_attention = EnhancedSpatialAttention()
        self.residual = residual
        
        # Class-aware learnable bias (3 classes: backpack, bag, trolley)
        self.class_bias = nn.Parameter(torch.tensor([0.1, 0.15, -0.1]))
        self.bias_proj = nn.Conv2d(c1, 3, 1, bias=False)
        self.bias_expand = nn.Conv2d(3, c1, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.channel_attention(x)
        out = self.spatial_attention(out)
        
        class_logits = self.bias_proj(out).mean(dim=(2, 3), keepdim=True)
        class_weights = torch.softmax(class_logits + self.class_bias.view(1, 3, 1, 1), dim=1)
        bias_adjustment = self.bias_expand(class_weights)
        out = out * (1 + 0.1 * bias_adjustment)
        
        if self.residual:
            out = out + identity
        return out


class LuggageCBAMv2(nn.Module):
    """
    LuggageCBAM v2 - Simplified and improved CBAM for luggage detection.
    
    Improvements over v1:
    1. Removed class-aware bias (let SATAL loss handle class balancing)
    2. Lower reduction ratio (8 instead of 16) - preserves more channel info
    3. Added scale-adaptive spatial attention (content-aware fusion)
    4. Learnable residual weight (α * attention + (1-α) * identity)
    5. Optional coordinate attention for better localization
    
    Paper contribution:
    "Simplified attention with content-aware multi-scale fusion"
    """
    
    # Class-level debug flag and counter
    _debug_enabled = False
    _debug_log_every = 100
    _debug_counter = 0
    _instance_id = 0

    def __init__(self, c1: int, reduction: int = 8, use_coord: bool = False):
        """
        Args:
            c1: Number of input channels
            reduction: Channel reduction ratio (default 8, less aggressive than v1's 16)
            use_coord: Whether to add coordinate attention (default False for speed)
        """
        super().__init__()
        
        # Instance identification for debug
        LuggageCBAMv2._instance_id += 1
        self.name = f"CBAMv2_{LuggageCBAMv2._instance_id}_c{c1}"
        
        # Channel attention with less aggressive reduction
        self.channel_attn = self._make_channel_attention(c1, reduction)
        
        # Scale-adaptive spatial attention
        self.spatial_attn = ScaleAdaptiveSpatialAttention()
        
        # Learnable residual weight (initialized to 0.5 = balanced)
        self.residual_weight = nn.Parameter(torch.tensor(0.5))
        
        # Optional coordinate attention for better localization
        self.use_coord = use_coord
        if use_coord:
            self.coord_attn = CoordinateAttention(c1, c1, reduction)
    
    @classmethod
    def enable_debug(cls, log_every: int = 100):
        """Enable debug logging for all instances."""
        cls._debug_enabled = True
        cls._debug_log_every = log_every
        cls._debug_counter = 0
        print(f"[LuggageCBAMv2] Debug enabled, logging every {log_every} batches")
    
    @classmethod
    def disable_debug(cls):
        """Disable debug logging."""
        cls._debug_enabled = False
    
    def _make_channel_attention(self, c1: int, reduction: int) -> nn.Module:
        """Create channel attention with both avg and max pooling."""
        c_mid = max(c1 // reduction, 16)  # At least 16 channels
        return nn.ModuleDict({
            'avg_pool': nn.AdaptiveAvgPool2d(1),
            'max_pool': nn.AdaptiveMaxPool2d(1),
            'mlp': nn.Sequential(
                nn.Conv2d(c1, c_mid, 1, bias=False),
                nn.SiLU(inplace=True),  # SiLU instead of ReLU (smoother gradients)
                nn.Conv2d(c_mid, c1, 1, bias=False),
            ),
            'act': nn.Sigmoid()
        })
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Channel attention
        avg_out = self.channel_attn['mlp'](self.channel_attn['avg_pool'](x))
        max_out = self.channel_attn['mlp'](self.channel_attn['max_pool'](x))
        channel_weight = self.channel_attn['act'](avg_out + max_out)
        out = x * channel_weight
        
        # Spatial attention (scale-adaptive)
        out, scale_weights = self.spatial_attn(out, return_weights=True)
        
        # Optional coordinate attention
        if self.use_coord:
            out = self.coord_attn(out)
        
        # Learnable residual connection
        alpha = torch.sigmoid(self.residual_weight)
        out = alpha * out + (1 - alpha) * identity
        
        # Debug logging
        if self._debug_enabled and self.training:
            LuggageCBAMv2._debug_counter += 1
            if LuggageCBAMv2._debug_counter % self._debug_log_every == 0:
                self._log_debug(channel_weight, scale_weights, alpha)
        
        return out
    
    def _log_debug(self, channel_weight: torch.Tensor, scale_weights: torch.Tensor, alpha: torch.Tensor):
        """Log debug information."""
        with torch.no_grad():
            # Channel attention stats
            ch_mean = channel_weight.mean().item()
            ch_std = channel_weight.std().item()
            ch_active = (channel_weight > 0.5).float().mean().item()
            
            # Scale selection stats
            sw = scale_weights.mean(dim=0).squeeze()
            w3, w7, w11 = sw[0].item(), sw[1].item(), sw[2].item()
            dominant = ['3x3', '7x7', '11x11'][sw.argmax().item()]
            
            # Residual weight
            alpha_val = alpha.item()
            
            print(f"[{self.name}] Ch: mean={ch_mean:.3f} active={ch_active:.1%} | "
                  f"Scale: 3x3={w3:.2f} 7x7={w7:.2f} 11x11={w11:.2f} ({dominant}) | "
                  f"α={alpha_val:.3f}")


class ScaleAdaptiveSpatialAttention(nn.Module):
    """
    Scale-Adaptive Spatial Attention for luggage detection.
    
    Key innovation: Content-aware fusion of multi-scale attention maps.
    Instead of fixed/single-parameter fusion, this learns to predict
    which scale is most relevant based on the input content.
    
    - 3x3 kernel: Fine details (straps, handles, small features)
    - 7x7 kernel: Medium context (bag body, backpack shape)
    - 11x11 kernel: Large context (trolley, full object)
    """

    def __init__(self):
        super().__init__()
        # Multi-scale spatial convolutions
        self.conv3 = nn.Conv2d(2, 1, 3, padding=1, bias=False)   # Fine details
        self.conv7 = nn.Conv2d(2, 1, 7, padding=3, bias=False)   # Medium context
        self.conv11 = nn.Conv2d(2, 1, 11, padding=5, bias=False) # Large context
        
        # Content-aware scale predictor
        # Predicts which scale to emphasize based on spatial statistics
        self.scale_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # Global context
            nn.Conv2d(2, 8, 1),       # Expand
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 3, 1),       # 3 scale weights
            nn.Softmax(dim=1)
        )
        
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, C, H, W]
            return_weights: If True, also return scale_weights for debugging
            
        Returns:
            out: Attended features [B, C, H, W]
            scale_weights: (optional) Scale selection weights [B, 3, 1, 1]
        """
        # Spatial features: channel avg + max
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
        # Multi-scale attention maps
        attn3 = self.conv3(spatial)    # Fine
        attn7 = self.conv7(spatial)    # Medium
        attn11 = self.conv11(spatial)  # Large
        
        # Content-aware scale weights [B, 3, 1, 1]
        scale_weights = self.scale_predictor(spatial)
        w3 = scale_weights[:, 0:1, :, :]
        w7 = scale_weights[:, 1:2, :, :]
        w11 = scale_weights[:, 2:3, :, :]
        
        # Weighted fusion of scales
        fused_attn = w3 * attn3 + w7 * attn7 + w11 * attn11
        
        out = x * self.act(fused_attn)
        
        if return_weights:
            return out, scale_weights
        return out


class CoordinateAttention(nn.Module):
    """
    Coordinate Attention (CVPR 2021) for better spatial localization.
    
    Captures long-range dependencies along H and W dimensions separately,
    then combines them. Good for detecting objects at various positions.
    """

    def __init__(self, c1: int, c2: int, reduction: int = 8):
        super().__init__()
        c_mid = max(c1 // reduction, 8)
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # [B, C, H, 1]
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # [B, C, 1, W]
        
        self.conv1 = nn.Conv2d(c1, c_mid, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.act = nn.SiLU(inplace=True)
        
        self.conv_h = nn.Conv2d(c_mid, c2, 1, bias=False)
        self.conv_w = nn.Conv2d(c_mid, c2, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Pool along H and W separately
        x_h = self.pool_h(x)  # [B, C, H, 1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, W, 1]
        
        # Concatenate and process
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        y = self.act(self.bn1(self.conv1(y)))
        
        # Split back
        x_h, x_w = torch.split(y, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        
        # Generate attention weights
        a_h = torch.sigmoid(self.conv_h(x_h))
        a_w = torch.sigmoid(self.conv_w(x_w))
        
        return x * a_h * a_w


class EMA(nn.Module):
    """
    Efficient Multi-scale Attention for small/medium object detection.
    
    Simpler and more effective than LuggageCBAM v1:
    - Parallel multi-scale convolutions (1x1, 3x3, 5x5) 
    - Shared channel attention with lower reduction
    - Coordinate attention for better spatial awareness
    - No class-aware bias (handled by loss function instead)
    """

    def __init__(self, c1: int, reduction: int = 8):
        """
        Args:
            c1: Number of input/output channels
            reduction: Channel reduction ratio (default: 8, less aggressive)
        """
        super().__init__()
        c_ = max(c1 // reduction, 8)
        
        # Multi-scale feature extraction (parallel branches)
        self.conv1x1 = nn.Conv2d(c1, c_, 1, bias=False)
        self.conv3x3 = nn.Conv2d(c1, c_, 3, padding=1, groups=c_, bias=False)  # depthwise
        self.conv5x5 = nn.Conv2d(c1, c_, 5, padding=2, groups=c_, bias=False)  # depthwise
        
        # Channel attention (shared, efficient)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c_ * 3, c_, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_, c1, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Coordinate attention for spatial awareness
        self.coord_h = nn.AdaptiveAvgPool2d((None, 1))
        self.coord_w = nn.AdaptiveAvgPool2d((1, None))
        self.coord_conv = nn.Sequential(
            nn.Conv2d(c1, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU(inplace=True)
        )
        self.coord_h_conv = nn.Conv2d(c_, c1, 1, bias=False)
        self.coord_w_conv = nn.Conv2d(c_, c1, 1, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale features
        f1 = self.conv1x1(x)
        f3 = self.conv3x3(x)
        f5 = self.conv5x5(x)
        
        # Channel attention from multi-scale
        f_cat = torch.cat([f1, f3, f5], dim=1)
        channel_attn = self.fc(self.pool(f_cat))
        
        # Coordinate attention
        h_pool = self.coord_h(x)  # [B, C, H, 1]
        w_pool = self.coord_w(x)  # [B, C, 1, W]
        
        # Combine h and w
        _, _, h, w = x.shape
        h_feat = self.coord_conv(h_pool)
        w_feat = self.coord_conv(w_pool.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        
        h_attn = torch.sigmoid(self.coord_h_conv(h_feat))
        w_attn = torch.sigmoid(self.coord_w_conv(w_feat))
        
        # Combine channel + coordinate attention
        out = x * channel_attn * h_attn * w_attn
        return out + x  # residual


class SimAM(nn.Module):
    """
    Simple parameter-free Attention Module.
    
    Based on "SimAM: A Simple, Parameter-Free Attention Module for CNNs"
    No learnable parameters - uses energy function to compute attention.
    Very lightweight and effective for small objects.
    """

    def __init__(self, c1: int = None, e_lambda: float = 1e-4):
        """
        Args:
            c1: Channels (unused, for API compatibility)
            e_lambda: Regularization parameter
        """
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        n = w * h - 1
        
        # Compute mean and variance
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        
        return x * torch.sigmoid(y)


class LSKA(nn.Module):
    """
    Large Separable Kernel Attention for luggage detection.
    
    Uses large kernels (up to 21x21) decomposed into depth-wise separable convolutions
    for capturing long-range dependencies needed for detecting various luggage sizes.
    """

    def __init__(self, c1: int, k_size: int = 7):
        """
        Args:
            c1: Number of input/output channels
            k_size: Kernel size for spatial attention (7, 11, 21, etc.)
        """
        super().__init__()
        
        # Decompose large kernel: k = (k//2)*2 + 1 
        # e.g., 21 = 10*2+1, so we use 1x11 + 11x1 + 1x11 + 11x1
        pad = k_size // 2
        
        # Channel mixing
        self.conv0 = nn.Conv2d(c1, c1, 1, bias=False)
        self.bn0 = nn.BatchNorm2d(c1)
        
        # Spatial: decomposed large kernel (horizontal + vertical)
        self.conv_h = nn.Conv2d(c1, c1, (1, k_size), padding=(0, pad), groups=c1, bias=False)
        self.conv_v = nn.Conv2d(c1, c1, (k_size, 1), padding=(pad, 0), groups=c1, bias=False)
        
        # Output
        self.conv1 = nn.Conv2d(c1, c1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel attention
        attn = self.act(self.bn0(self.conv0(x)))
        
        # Spatial attention with large kernel
        attn = self.conv_h(attn)
        attn = self.conv_v(attn)
        attn = self.bn1(self.conv1(attn))
        
        return x * torch.sigmoid(attn)


class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v2 wrapper with fallback.
    
    Uses torchvision.ops.deform_conv2d if available, otherwise falls back
    to a simpler offset-modulated convolution.
    """
    
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, p: int = 1, g: int = 1):
        """
        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
            s: Stride
            p: Padding
            g: Groups
        """
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.k = k
        self.s = s
        self.p = p
        self.g = g
        
        # Check if torchvision deform_conv2d is available and working
        self.use_torchvision = False
        try:
            from torchvision.ops import deform_conv2d
            # Test if it works
            _test = torch.zeros(1, c1, 8, 8)
            _off = torch.zeros(1, 2 * k * k, 8, 8)
            _w = torch.zeros(c2, c1 // g, k, k)
            deform_conv2d(_test, _off, _w, stride=s, padding=p)
            self.use_torchvision = True
        except Exception:
            self.use_torchvision = False
        
        if self.use_torchvision:
            # Offset conv: predicts 2*k*k offsets (x,y for each kernel position)
            self.offset_conv = nn.Conv2d(c1, 2 * k * k, k, s, p, bias=True)
            # Modulation conv: predicts k*k modulation scalars
            self.modulator_conv = nn.Conv2d(c1, k * k, k, s, p, bias=True)
            # Main conv weights
            self.weight = nn.Parameter(torch.empty(c2, c1 // g, k, k))
            self.bias = nn.Parameter(torch.zeros(c2))
            
            # Initialize
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            nn.init.zeros_(self.offset_conv.weight)
            nn.init.zeros_(self.offset_conv.bias)
            nn.init.zeros_(self.modulator_conv.weight)
            nn.init.constant_(self.modulator_conv.bias, 0.5)
        else:
            # Fallback: Adaptive kernel attention (no deformation, but shape-aware)
            # Uses dynamic convolution with content-adaptive weights
            self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=True)
            self.offset_pred = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(c1, c1 // 4, 1),
                nn.SiLU(inplace=True),
                nn.Conv2d(c1 // 4, k * k, 1),
                nn.Sigmoid()
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_torchvision:
            from torchvision.ops import deform_conv2d
            offset = self.offset_conv(x)
            modulator = torch.sigmoid(self.modulator_conv(x))
            return deform_conv2d(
                x, offset, self.weight, self.bias,
                stride=self.s, padding=self.p, mask=modulator
            )
        else:
            # Fallback: content-adaptive convolution
            # Predict spatial importance weights based on content
            b, c, h, w = x.shape
            weights = self.offset_pred(x)  # [B, k*k, 1, 1]
            out = self.conv(x)
            # Modulate output based on learned content weights
            return out * weights.mean(dim=1, keepdim=True).expand_as(out)


class AdaptiveShapeConv(nn.Module):
    """
    Adaptive Shape Convolution - learns shape-aware attention without deformable ops.
    
    Uses separate horizontal and vertical convolutions with learned mixing weights
    to adapt to different object shapes (tall trolleys vs wide bags).
    
    More stable alternative to DeformableConv2d.
    """
    
    def __init__(self, c1: int, c2: int, k: int = 7):
        """
        Args:
            c1: Input channels
            c2: Output channels
            k: Kernel size
        """
        super().__init__()
        # Horizontal kernel for wide objects (bags)
        self.conv_h = nn.Conv2d(c1, c2, (1, k), padding=(0, k//2), bias=False)
        # Vertical kernel for tall objects (trolleys)
        self.conv_v = nn.Conv2d(c1, c2, (k, 1), padding=(k//2, 0), bias=False)
        # Square kernel for uniform objects (backpacks)
        self.conv_s = nn.Conv2d(c1, c2, 3, padding=1, bias=False)
        
        # Shape predictor - predicts mixing weights based on content
        self.shape_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c1, c1 // 4, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c1 // 4, 3, 1),  # 3 weights: horizontal, vertical, square
            nn.Softmax(dim=1)
        )
        
        self.bn = nn.BatchNorm2d(c2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predict shape weights
        weights = self.shape_pred(x)  # [B, 3, 1, 1]
        w_h, w_v, w_s = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        
        # Apply shape-specific convolutions
        out_h = self.conv_h(x)
        out_v = self.conv_v(x)
        out_s = self.conv_s(x)
        
        # Mix based on predicted shape
        out = w_h * out_h + w_v * out_v + w_s * out_s
        return self.bn(out)


class DeformableSpatialAttention(nn.Module):
    """
    Deformable Spatial Attention - learns to attend to object shapes.
    
    Unlike fixed kernel attention, this adapts sampling locations to:
    - Tall objects (trolleys): sample vertically
    - Wide objects (bags): sample horizontally  
    - Square objects (backpacks): sample uniformly
    """
    
    def __init__(self, c1: int):
        """
        Args:
            c1: Number of input channels
        """
        super().__init__()
        # Deformable conv for shape-adaptive spatial attention
        self.deform_conv = DeformableConv2d(2, 1, k=7, s=1, p=3)
        self.act = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Channel-wise statistics (same as standard spatial attention)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        
        # Deformable attention - adapts to object shape
        attn = self.deform_conv(spatial_features)
        
        return x * self.act(attn)


class DCBAM(nn.Module):
    """
    Deformable CBAM (D-CBAM) for shape-adaptive luggage detection.
    
    Key innovation: Replaces fixed spatial convolutions with deformable convolutions
    that learn to adapt their sampling pattern to object shapes:
    - Trolleys (tall, narrow): vertical sampling pattern
    - Bags (wide, flat): horizontal sampling pattern
    - Backpacks (square): uniform sampling pattern
    
    Paper contribution: "Shape-adaptive attention complements size-adaptive loss (SATAL)"
    
    Components:
    1. Enhanced Channel Attention (same as LuggageCBAM)
    2. Deformable Spatial Attention (NEW - shape-adaptive)
    3. Residual connection
    """
    
    def __init__(self, c1: int, reduction: int = 16, residual: bool = True):
        """
        Args:
            c1: Number of input/output channels
            reduction: Channel reduction ratio for channel attention
            residual: Whether to use residual connection
        """
        super().__init__()
        self.channel_attention = EnhancedChannelAttention(c1, reduction)
        self.spatial_attention = DeformableSpatialAttention(c1)
        self.residual = residual
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Channel attention (which channels to focus on)
        out = self.channel_attention(x)
        
        # Deformable spatial attention (where to focus, adapts to shape)
        out = self.spatial_attention(out)
        
        # Residual connection
        if self.residual:
            out = out + identity
            
        return out


class DCBAM_MS(nn.Module):
    """
    Multi-Scale Deformable CBAM (D-CBAM-MS).
    
    Combines deformable attention at multiple scales for detecting
    luggage of all sizes (small 27%, medium 57%, large 16%).
    
    Uses 3x3 deformable for fine details + 7x7 deformable for context.
    """
    
    def __init__(self, c1: int, reduction: int = 16, residual: bool = True):
        """
        Args:
            c1: Number of input/output channels
            reduction: Channel reduction ratio
            residual: Whether to use residual connection
        """
        super().__init__()
        self.channel_attention = EnhancedChannelAttention(c1, reduction)
        
        # Multi-scale deformable spatial attention
        self.deform_small = DeformableConv2d(2, 1, k=3, s=1, p=1)  # Fine details
        self.deform_large = DeformableConv2d(2, 1, k=7, s=1, p=3)  # Context
        
        # Learnable scale fusion
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.act = nn.Sigmoid()
        self.residual = residual
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Channel attention
        out = self.channel_attention(x)
        
        # Spatial features
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out = torch.max(out, dim=1, keepdim=True)[0]
        spatial_features = torch.cat([avg_out, max_out], dim=1)
        
        # Multi-scale deformable attention
        attn_small = self.deform_small(spatial_features)
        attn_large = self.deform_large(spatial_features)
        
        # Learnable fusion
        w = torch.sigmoid(self.fusion_weight)
        fused_attn = w * attn_small + (1 - w) * attn_large
        
        out = out * self.act(fused_attn)
        
        if self.residual:
            out = out + identity
            
        return out


class ShapeSpatialAttention(nn.Module):
    """
    Shape-Adaptive Spatial Attention (no deformable conv - stable alternative).
    
    Uses separate H/V/Square convolutions mixed based on content to adapt
    to different luggage shapes without requiring deformable convolutions.
    
    Key innovation: Learns to predict which kernel shape (H/V/Square) is best
    for each input based on spatial statistics, then mixes attention maps accordingly.
    """
    
    def __init__(self, k: int = 7):
        super().__init__()
        # Shape-specific convolutions on spatial features (2 channels: avg + max)
        self.conv_h = nn.Conv2d(2, 1, (1, k), padding=(0, k//2), bias=False)  # Horizontal (bags)
        self.conv_v = nn.Conv2d(2, 1, (k, 1), padding=(k//2, 0), bias=False)  # Vertical (trolleys)
        self.conv_s = nn.Conv2d(2, 1, 3, padding=1, bias=False)               # Square (backpacks)
        
        # Shape predictor - uses Conv2d instead of Linear for proper spatial pooling
        # Input: spatial features [B, 2, H, W] -> Output: shape weights [B, 3, 1, 1]
        self.shape_pred = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 2, H, W] -> [B, 2, 1, 1]
            nn.Conv2d(2, 3, 1),       # [B, 2, 1, 1] -> [B, 3, 1, 1]
            nn.Softmax(dim=1)
        )
        self.act = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial features: combine channel avg and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, H, W]
        spatial = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        
        # Shape-specific attention maps
        attn_h = self.conv_h(spatial)  # Horizontal kernel for wide objects (bags)
        attn_v = self.conv_v(spatial)  # Vertical kernel for tall objects (trolleys)
        attn_s = self.conv_s(spatial)  # Square kernel for uniform objects (backpacks)
        
        # Predict shape weights based on spatial statistics [B, 3, 1, 1]
        weights = self.shape_pred(spatial)
        w_h = weights[:, 0:1, :, :]  # [B, 1, 1, 1]
        w_v = weights[:, 1:2, :, :]  # [B, 1, 1, 1]
        w_s = weights[:, 2:3, :, :]  # [B, 1, 1, 1]
        
        # Mix attention maps based on predicted shape
        attn = w_h * attn_h + w_v * attn_v + w_s * attn_s
        
        return x * self.act(attn)


class ShapeCBAM(nn.Module):
    """
    Shape-Adaptive CBAM (S-CBAM) for luggage detection.
    
    STABLE ALTERNATIVE to D-CBAM (no deformable convolutions).
    
    Key innovation: Uses shape-specific convolutions (horizontal, vertical, square)
    mixed dynamically based on content to adapt to different luggage shapes:
    - Trolleys (tall): emphasizes vertical convolution
    - Bags (wide): emphasizes horizontal convolution
    - Backpacks (square): emphasizes square convolution
    
    Paper contribution: "Shape-adaptive attention complements size-adaptive loss (SATAL)"
    """
    
    def __init__(self, c1: int, reduction: int = 16, residual: bool = True):
        """
        Args:
            c1: Number of input/output channels
            reduction: Channel reduction ratio for channel attention
            residual: Whether to use residual connection
        """
        super().__init__()
        self.channel_attention = EnhancedChannelAttention(c1, reduction)
        self.spatial_attention = ShapeSpatialAttention(k=7)
        self.residual = residual
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        # Channel attention
        out = self.channel_attention(x)
        
        # Shape-adaptive spatial attention
        out = self.spatial_attention(out)
        
        # Residual
        if self.residual:
            out = out + identity
            
        return out


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input."""

    def __init__(self, c1, c2, index=0):
        """Returns a particular index of the input."""
        super().__init__()
        self.index = index

    def forward(self, x):
        """
        Forward pass.

        Expects a list of tensors as input.
        """
        return x[self.index]


class SmallObjectRefinement(nn.Module):
    """
    Multi-scale dilated context + coordinate attention for small-object detection.
    
    Designed for P3 detection feature only.
    Combines three receptive fields (1, 2, 3 dilation) for scale-robust small features
    plus coordinate attention for precise localization.
    
    Key differences from other attention modules:
    - CBAM: channel + spatial attention (single receptive field)
    - ChannelSE: channel attention only
    - LSKA: large kernel attention (single receptive field)
    - SmallObjectRefinement: MULTI receptive fields + coordinate attention
    
    For varied-size small objects (range 11-1024 px², median ~49×49), 
    multi-RF features are more robust than single fixed kernel.
    """
    
    def __init__(self, c1, c2=None):
        """
        Args:
            c1: Input channels
            c2: Output channels (defaults to c1)
        """
        super().__init__()
        c2 = c2 or c1
        c_ = c1 // 4  # intermediate channel size
        
        # Three parallel branches at different dilations for multi-scale context
        self.b1 = Conv(c1, c_, 3, 1, d=1)  # dilation 1: local context (RF=3)
        self.b2 = Conv(c1, c_, 3, 1, d=2)  # dilation 2: mid-range context (RF=5)
        self.b3 = Conv(c1, c_, 3, 1, d=3)  # dilation 3: extended context (RF=7)
        self.b4 = Conv(c1, c_, 1, 1)       # 1x1 identity branch (preserves fine details)
        
        # Coordinate attention (cheap, position-aware)
        # Captures H and W positional information separately
        self.coord_h = nn.AdaptiveAvgPool2d((None, 1))  # [B, C, H, 1]
        self.coord_w = nn.AdaptiveAvgPool2d((1, None))  # [B, C, 1, W]
        self.coord_conv = nn.Sequential(
            nn.Conv2d(c_ * 4, c_, 1, bias=False),
            nn.BatchNorm2d(c_),
            nn.SiLU()
        )
        self.coord_out_h = nn.Conv2d(c_, c2, 1)
        self.coord_out_w = nn.Conv2d(c_, c2, 1)
        
        # Fusion layer
        self.fuse = Conv(c_ * 4, c2, 1, 1)
        
        # Match input/output channels for residual
        self.shortcut = nn.Identity() if c1 == c2 else Conv(c1, c2, 1, 1, act=False)
        
    def forward(self, x):
        """Forward pass with multi-dilation context and coordinate attention."""
        identity = self.shortcut(x)
        
        # Multi-dilation context aggregation
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        fused = self.fuse(y)
        
        # Coordinate attention from fused features
        h_pool = self.coord_h(y)  # [B, c_*4, H, 1]
        w_pool = self.coord_w(y)  # [B, c_*4, 1, W]
        
        # Process H and W coordinates
        h_feat = self.coord_conv(h_pool)  # [B, c_, H, 1]
        w_feat = self.coord_conv(w_pool.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)  # [B, c_, 1, W]
        
        # Generate attention maps
        h_attn = torch.sigmoid(self.coord_out_h(h_feat))  # [B, c2, H, 1]
        w_attn = torch.sigmoid(self.coord_out_w(w_feat))  # [B, c2, 1, W]
        
        # Apply coordinate attention and residual connection
        return fused * h_attn * w_attn + identity
