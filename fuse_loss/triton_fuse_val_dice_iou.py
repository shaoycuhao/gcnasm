#Only tested with binary segmentation tasks
#Dice and IOU(jaccard also) evaluation fused for CV tasks especially with diagnostic uses
import triton
import triton.language as tl
import torch

@triton.jit
def compute_metrics_kernel(
    logits_ptr, targets_ptr, stats_ptr,
    B, C, H, W,
    stride_b, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
    NUM_CLASSES: tl.constexpr,
    IGNORE_INDEX: tl.constexpr = -100,
):

    b = tl.program_id(0)
    total_pixels = H * W

    for base in range(0, total_pixels, BLOCK_SIZE):
        offsets = base + tl.arange(0, BLOCK_SIZE)
        mask = offsets < total_pixels

        h = offsets // W
        w = offsets % W

        target = tl.load(targets_ptr + b * H * W + offsets, mask=mask)

        max_logit = tl.full(mask.shape, -1e10, dtype=tl.float32)
        pred = tl.full(mask.shape, 0, dtype=tl.int32)

        for c in range(NUM_CLASSES):
            off = b * stride_b + c * stride_c + h * stride_h + w * stride_w
            val = tl.load(logits_ptr + off, mask=mask)

            val = tl.where(mask, val, -1e10)
            is_max = val > max_logit
            max_logit = tl.where(is_max, val, max_logit)
            pred = tl.where(is_max, c, pred)

        valid = (target != IGNORE_INDEX) & mask

        tp_cond = (pred == target) & valid
        fp_cond = (pred != target) & valid
        fn_cond = fp_cond   
        add_tp = tl.where(tp_cond, 1.0, 0.0)
        add_fp = tl.where(fp_cond, 1.0, 0.0)
        add_fn = tl.where(fn_cond, 1.0, 0.0)

        stats_base = stats_ptr + b * NUM_CLASSES * 3

        tl.atomic_add(stats_base + pred * 3 + 0, add_tp)
        tl.atomic_add(stats_base + pred * 3 + 1, add_fp)
        tl.atomic_add(stats_base + target * 3 + 2, add_fn)


def compute_metrics(logits, targets, num_classes, ignore_index=-100):
    B, C, H, W = logits.shape
    assert C == num_classes, f"logits channels {C} != num_classes {num_classes}"
    device = logits.device
    targets = targets.long().contiguous()

    stats = torch.zeros((B, num_classes, 3), dtype=torch.float32, device=device)

    stride_b, stride_c, stride_h, stride_w = logits.stride()
    BLOCK_SIZE = 256          
    grid = (B,)              

    compute_metrics_kernel[grid](
        logits, targets, stats,
        B, C, H, W,
        stride_b, stride_c, stride_h, stride_w,
        BLOCK_SIZE=BLOCK_SIZE,
        NUM_CLASSES=num_classes,
        IGNORE_INDEX=ignore_index,
    )

    tp = stats[:, :, 0]   
    fp = stats[:, :, 1]
    fn = stats[:, :, 2]

    denom = 2 * tp + fp + fn
    dice_per_class = torch.where(denom == 0, torch.ones_like(denom), (2 * tp) / denom)

    denom_iou = tp + fp + fn
    iou_per_class = torch.where(denom_iou == 0, torch.ones_like(denom_iou), tp / denom_iou)

    sample_dice = dice_per_class.mean(dim=1)   # (B,)
    sample_iou  = iou_per_class.mean(dim=1)    # (B,)

    return sample_dice, sample_iou