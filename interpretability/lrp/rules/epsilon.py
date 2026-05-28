import torch
import torch.nn.functional as F

import lxt.explicit.functional as lf


def _eps(x, epsilon=1e-6):
    """Add ε with the same sign as x (prevents relevance flip at zero crossings)."""
    return x + epsilon * x.sign().where(x != 0, torch.ones_like(x))


# ─────────────────────────────────────────────────────────────────────────────
# Conv2d ε-rule
# ─────────────────────────────────────────────────────────────────────────────


class Conv2dEpsilonFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, stride, padding, dilation, groups, epsilon):
        z = F.conv2d(x, weight, bias, stride, padding, dilation, groups)
        ctx.save_for_backward(x, weight, z)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.epsilon = epsilon
        return z

    @staticmethod
    def backward(ctx, out_relevance):
        x, weight, z = ctx.saved_tensors
        relevance_norm = out_relevance / _eps(z, ctx.epsilon)

        kH, kW = weight.shape[2], weight.shape[3]
        op_h = x.shape[2] - (
            (z.shape[2] - 1) * ctx.stride[0]
            - 2 * ctx.padding[0]
            + ctx.dilation[0] * (kH - 1)
            + 1
        )
        op_w = x.shape[3] - (
            (z.shape[3] - 1) * ctx.stride[1]
            - 2 * ctx.padding[1]
            + ctx.dilation[1] * (kW - 1)
            + 1
        )

        grad = F.conv_transpose2d(
            relevance_norm, weight, bias=None,
            stride=ctx.stride, padding=ctx.padding,
            output_padding=(op_h, op_w),
            dilation=ctx.dilation, groups=ctx.groups,
        )
        return grad * x, None, None, None, None, None, None, None


def conv2d_epsilon(x, weight, bias, stride=(1, 1), padding=(0, 0),
                   dilation=(1, 1), groups=1, epsilon=0.001):
    """ε-rule for nn.Conv2d. Drop-in for F.conv2d in forward methods."""
    stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
    padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    dilation = (dilation, dilation) if isinstance(dilation, int) else tuple(dilation)
    if x.dtype != weight.dtype:
        x = x.to(weight.dtype)
    return Conv2dEpsilonFn.apply(x, weight, bias, stride, padding, dilation, groups, epsilon)


# ─────────────────────────────────────────────────────────────────────────────
# Elementwise scale ε-rule  (z = scale * x, scale is a parameter)
# ─────────────────────────────────────────────────────────────────────────────


class ScaleEpsilonFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, scale, epsilon):
        z = scale * x
        ctx.save_for_backward(x, scale, z)
        ctx.epsilon = epsilon
        return z

    @staticmethod
    def backward(ctx, R_out):
        x, scale, z = ctx.saved_tensors
        # ε-rule
        R_norm = R_out / _eps(z, ctx.epsilon)
        return R_norm * scale * x, None, None


def scale_epsilon(x, scale, epsilon=0.001):
    """ε-rule for elementwise z = scale * x where scale is a learned parameter."""
    return ScaleEpsilonFn.apply(x, scale, epsilon)


# ─────────────────────────────────────────────────────────────────────────────
# AvgPool2d ε-rule
# ─────────────────────────────────────────────────────────────────────────────


class AvgPool2dEpsilonFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, kernel_size, stride, padding, epsilon):
        z = F.avg_pool2d(x, kernel_size, stride, padding)
        ctx.save_for_backward(x, z)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        ctx.epsilon = epsilon
        return z

    @staticmethod
    def backward(ctx, R_out):
        x, z = ctx.saved_tensors
        R_norm = R_out / _eps(z, ctx.epsilon)

        kH, kW = ctx.kernel_size
        sH, sW = ctx.stride
        pH, pW = ctx.padding
        N = kH * kW
        C = x.shape[1]

        # Depthwise transposed conv with uniform 1/N weights recovers the
        # avg_pool gradient (each input receives 1/N of each output it contributed to).
        weight = torch.ones(C, 1, kH, kW, dtype=R_norm.dtype, device=R_norm.device) / N
        oH = (z.shape[2] - 1) * sH - 2 * pH + kH
        oW = (z.shape[3] - 1) * sW - 2 * pW + kW
        op_h = x.shape[2] - oH
        op_w = x.shape[3] - oW
        grad = F.conv_transpose2d(
            R_norm, weight, bias=None,
            stride=ctx.stride, padding=ctx.padding,
            output_padding=(op_h, op_w), groups=C,
        )
        return grad * x, None, None, None, None


def avgpool2d_epsilon(x, kernel_size, stride=None, padding=0, epsilon=0.001):
    """ε-rule for nn.AvgPool2d. Propagates relevance proportionally to each input's value."""
    if stride is None:
        stride = kernel_size
    kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
    stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
    padding = (padding, padding) if isinstance(padding, int) else tuple(padding)
    return AvgPool2dEpsilonFn.apply(x, kernel_size, stride, padding, epsilon)


# ─────────────────────────────────────────────────────────────────────────────
# GroupNorm ε-rule
# ─────────────────────────────────────────────────────────────────────────────


class GroupNormGradFn(torch.autograd.Function):
    """
    Proper ε-rule for GroupNorm, mirroring LXT's layer_norm_grad_fn approach.

    Forward: compute GroupNorm with std detached (identity rule on 1/std).
    Backward: R_x = grad(y, x, R_out / (y + ε)) * x  (ε-rule, conserves relevance).
    """

    @staticmethod
    def forward(ctx, x, num_groups, weight, bias, eps, epsilon=1e-6):
        with torch.enable_grad():
            N_ = x.shape[0]
            C = x.shape[1]
            G = num_groups
            spatial = x.shape[2:]
            x_g = x.view(N_, G, C // G, *spatial)
            dims = list(range(2, x_g.dim()))
            mean = x_g.mean(dim=dims, keepdim=True)
            var = ((x_g - mean) ** 2).mean(dim=dims, keepdim=True)
            std = (var + eps).sqrt()
            y = ((x_g - mean) / std.detach()).view_as(x)
            if weight is not None:
                y = y * weight.view(1, -1, *([1] * len(spatial)))
            if bias is not None:
                y = y + bias.view(1, -1, *([1] * len(spatial)))

            ctx.save_for_backward(x, y)
            ctx.epsilon = epsilon

        return y.detach()

    @staticmethod
    def backward(ctx, *out_relevance):
        x, y = ctx.saved_tensors
        epsilon = ctx.epsilon

        relevance_norm = out_relevance[0] / (y + epsilon * y.sign().where(y != 0, torch.ones_like(y)))

        grads, = torch.autograd.grad(y, x, relevance_norm)

        return (grads * x, None, None, None, None, None)


# ─────────────────────────────────────────────────────────────────────────────
# Module forward-method patches (bound as methods via types.MethodType).
# Bridge the (self, x) method signature to the functional ε-rules above
# (or to LXT's functional equivalents).
# ─────────────────────────────────────────────────────────────────────────────


def linear_lrp_forward(self, x):
    """Drop-in for nn.Linear.forward using LXT ε-rule."""
    return lf.linear_epsilon(x, self.weight, self.bias)


def layernorm_lrp_forward(self, x):
    """Drop-in for nn.LayerNorm.forward using LXT layer_norm."""
    return lf.layer_norm(x, self.weight, self.bias, self.eps)


def rmsnorm_lrp_forward(self, x):
    """Drop-in for RMSNorm.forward using LXT rms_norm_identity."""
    eps = self.eps
    if eps is None:
        eps = torch.finfo(x.dtype).eps
    return lf.rms_norm_identity(x, self.weight, eps)


def groupnorm_lrp_forward(self, x):
    """ε-rule for nn.GroupNorm via GroupNormGradFn."""
    return GroupNormGradFn.apply(x, self.num_groups, self.weight, self.bias, self.eps)


def conv2d_epsilon_forward(self, x):
    """ε-rule replacement for nn.Conv2d.forward."""
    return conv2d_epsilon(
        x, self.weight, self.bias,
        stride=self.stride, padding=self.padding,
        dilation=self.dilation, groups=self.groups,
    )


def avgpool2d_epsilon_forward(self, x):
    """ε-rule replacement for nn.AvgPool2d.forward."""
    return avgpool2d_epsilon(
        x, self.kernel_size, self.stride, self.padding,
    )

