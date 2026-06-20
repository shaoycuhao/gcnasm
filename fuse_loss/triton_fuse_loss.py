#Only for binary segmentation tasks
#CrossEntrophy and Dice Loss fused for CV tasks especially with diagnostic uses
import torch
import triton
import triton.language as tl
from torch.autograd import Function


SMOOTH = 1e-5
BLOCK_SIZE = 256         

@triton.jit
def forward_kernel(
    logits_ptr, targets_ptr, aux_ptr,
    B, H, W,
    stride_b, stride_c, stride_h, stride_w,
    SMOOTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < B * H * W


    b = offsets // (H * W)
    hw = offsets % (H * W)
    h = hw // W
    w = hw % W

    off0 = b * stride_b + 0 * stride_c + h * stride_h + w * stride_w
    off1 = b * stride_b + 1 * stride_c + h * stride_h + w * stride_w

    l0 = tl.load(logits_ptr + off0, mask=mask)
    l1 = tl.load(logits_ptr + off1, mask=mask)

    max_l = tl.maximum(l0, l1)
    e0 = tl.exp(l0 - max_l)
    e1 = tl.exp(l1 - max_l)
    sum_e = e0 + e1
    p0 = e0 / sum_e
    p1 = e1 / sum_e

    target = tl.load(targets_ptr + offsets, mask=mask)  # [BLOCK_SIZE]


    ce = tl.where(target == 0, -tl.log(p0), -tl.log(p1))
    inter0 = tl.where(target == 0, p0, 0.0)
    union0 = tl.where(target == 0, p0 + 1.0, 0.0)
    inter1 = tl.where(target == 1, p1, 0.0)
    union1 = tl.where(target == 1, p1 + 1.0, 0.0)


    ce_sum = tl.sum(ce, axis=0)
    inter0_sum = tl.sum(inter0, axis=0)
    union0_sum = tl.sum(union0, axis=0)
    inter1_sum = tl.sum(inter1, axis=0)
    union1_sum = tl.sum(union1, axis=0)
    count = tl.sum(mask.to(tl.float32), axis=0)


    tl.atomic_add(aux_ptr + 0, ce_sum)
    tl.atomic_add(aux_ptr + 1, inter0_sum)
    tl.atomic_add(aux_ptr + 2, union0_sum)
    tl.atomic_add(aux_ptr + 3, inter1_sum)
    tl.atomic_add(aux_ptr + 4, union1_sum)
    tl.atomic_add(aux_ptr + 5, count)

@triton.jit
def backward_kernel(
    logits_ptr, targets_ptr, grad_logits_ptr, aux_ptr,
    B, H, W,
    stride_b, stride_c, stride_h, stride_w,
    SMOOTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

    inter0 = tl.load(aux_ptr + 1)
    union0 = tl.load(aux_ptr + 2)
    inter1 = tl.load(aux_ptr + 3)
    union1 = tl.load(aux_ptr + 4)
    count = tl.load(aux_ptr + 5)


    denom0 = (union0 + SMOOTH) * (union0 + SMOOTH)
    grad_dice0_y1 = -(2.0 * (union0 + SMOOTH) - (2.0 * inter0 + SMOOTH)) / denom0 / 2.0
    grad_dice0_y0 = -(-(2.0 * inter0 + SMOOTH)) / denom0 / 2.0
    denom1 = (union1 + SMOOTH) * (union1 + SMOOTH)
    grad_dice1_y1 = -(2.0 * (union1 + SMOOTH) - (2.0 * inter1 + SMOOTH)) / denom1 / 2.0
    grad_dice1_y0 = -(-(2.0 * inter1 + SMOOTH)) / denom1 / 2.0

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < B * H * W

    b = offsets // (H * W)
    hw = offsets % (H * W)
    h = hw // W
    w = hw % W

    off0 = b * stride_b + 0 * stride_c + h * stride_h + w * stride_w
    off1 = b * stride_b + 1 * stride_c + h * stride_h + w * stride_w

    l0 = tl.load(logits_ptr + off0, mask=mask)
    l1 = tl.load(logits_ptr + off1, mask=mask)

    max_l = tl.maximum(l0, l1)
    e0 = tl.exp(l0 - max_l)
    e1 = tl.exp(l1 - max_l)
    sum_e = e0 + e1
    p0 = e0 / sum_e
    p1 = e1 / sum_e

    target = tl.load(targets_ptr + offsets, mask=mask)


    grad_ce_0 = tl.where(target == 0, (p0 - 1.0) / count, p0 / count)
    grad_ce_1 = tl.where(target == 0, p1 / count, (p1 - 1.0) / count)


    dice_grad0 = tl.where(target == 0, grad_dice0_y1, grad_dice0_y0)
    dice_grad1 = tl.where(target == 0, grad_dice1_y0, grad_dice1_y1)

    grad0 = grad_ce_0 + dice_grad0
    grad1 = grad_ce_1 + dice_grad1

    tl.store(grad_logits_ptr + off0, grad0, mask=mask)
    tl.store(grad_logits_ptr + off1, grad1, mask=mask)

class FusedCEDiceLoss(Function):
    @staticmethod
    def forward(ctx, logits, targets):
        B, C, H, W = logits.shape
        assert C == 2, "Only binary segmentation supported"
        device = logits.device
        aux = torch.zeros(6, dtype=torch.float32, device=device)

        stride_b, stride_c, stride_h, stride_w = logits.stride()
        total_pixels = B * H * W
        grid = ( (total_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE, )

        forward_kernel[grid](
            logits, targets, aux,
            B, H, W,
            stride_b, stride_c, stride_h, stride_w,
            SMOOTH,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ce_sum = aux[0].item()
        inter0 = aux[1].item()
        union0 = aux[2].item()
        inter1 = aux[3].item()
        union1 = aux[4].item()
        count = aux[5].item()

        ce_loss = ce_sum / count
        dice0 = (2.0 * inter0 + SMOOTH) / (union0 + SMOOTH)
        dice1 = (2.0 * inter1 + SMOOTH) / (union1 + SMOOTH)
        dice_loss = 1.0 - (dice0 + dice1) / 2.0
        loss = ce_loss + dice_loss


        loss_tensor = torch.tensor(loss, dtype=torch.float32, device=device)

        ctx.save_for_backward(logits, targets, aux)
        return loss_tensor

    @staticmethod
    def backward(ctx, grad_output):
        logits, targets, aux = ctx.saved_tensors
        B, C, H, W = logits.shape
        device = logits.device
        grad_logits = torch.zeros_like(logits)

        stride_b, stride_c, stride_h, stride_w = logits.stride()
        total_pixels = B * H * W
        grid = ( (total_pixels + BLOCK_SIZE - 1) // BLOCK_SIZE, )

        backward_kernel[grid](
            logits, targets, grad_logits, aux,
            B, H, W,
            stride_b, stride_c, stride_h, stride_w,
            SMOOTH,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        if grad_output != 1.0:
            grad_logits = grad_logits * grad_output
        return grad_logits, None