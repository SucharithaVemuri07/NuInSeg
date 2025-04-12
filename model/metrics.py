import numpy as np
from scipy.optimize import linear_sum_assignment
import torch

#As per MoNuSeg distributed metrics - https://github.com/vqdang/hover_net.git
def dice_metric(pred, target, smooth=1e-6):
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = (pred * target).sum()
    return (2 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def aji_fast_metric(gt, pred):
    """Calculates the fast AJI score between ground truth and predicted masks.

    Args:
        gt (numpy.ndarray): Ground truth instance mask (assumed remapped to [1,2,3,...]).
        pred (numpy.ndarray): Predicted instance mask (assumed remapped to [1,2,3,...]).

    Returns:
        float: AJI score.
    """
    def compute_single(gt_mask, pred_mask):
        true_ids = np.unique(gt_mask)
        pred_ids = np.unique(pred_mask)

        true_ids = true_ids[true_ids != 0]
        pred_ids = pred_ids[pred_ids != 0]

        true_masks = [None] + [(gt_mask == t).astype(np.uint8) for t in true_ids]
        pred_masks = [None] + [(pred_mask == p).astype(np.uint8) for p in pred_ids]

        pairwise_inter = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float64)
        pairwise_union = np.zeros((len(true_ids), len(pred_ids)), dtype=np.float64)

        for t_idx, t_id in enumerate(true_ids, 1):
            t_mask = true_masks[t_idx]
            overlapping_preds = np.unique(pred_mask[t_mask > 0])
            overlapping_preds = overlapping_preds[overlapping_preds != 0]
            for p_id in overlapping_preds:
                p_mask = pred_masks[p_id]
                inter = np.logical_and(t_mask, p_mask).sum()
                union = np.logical_or(t_mask, p_mask).sum()
                pairwise_inter[t_idx-1, p_id-1] = inter
                pairwise_union[t_idx-1, p_id-1] = union

        pairwise_iou = pairwise_inter / (pairwise_union + 1e-6)

        paired_pred = np.argmax(pairwise_iou, axis=1)
        max_iou = np.max(pairwise_iou, axis=1)
        paired_true = np.where(max_iou > 0.0)[0]

        overall_inter = pairwise_inter[paired_true, paired_pred[paired_true]].sum()
        overall_union = pairwise_union[paired_true, paired_pred[paired_true]].sum()

        paired_true_ids = set(true_ids[paired_true])
        paired_pred_ids = set(pred_ids[paired_pred[paired_true]])

        unpaired_true = [t for t in true_ids if t not in paired_true_ids]
        unpaired_pred = [p for p in pred_ids if p not in paired_pred_ids]

        for t_id in unpaired_true:
            overall_union += true_masks[true_ids.tolist().index(t_id) + 1].sum()
        for p_id in unpaired_pred:
            overall_union += pred_masks[pred_ids.tolist().index(p_id) + 1].sum()

        return overall_inter / (overall_union + 1e-6)

    if gt.ndim == 3:
        scores = []
        for i in range(gt.shape[0]):
            scores.append(compute_single(gt[i], pred[i]))
        return np.mean(scores)
    else:
        return compute_single(gt, pred)

def pq_fast_metric(true, pred, match_iou=0.5):

    true = np.copy(true)
    pred = np.copy(pred)
    true_id_list = list(np.unique(true))
    pred_id_list = list(np.unique(pred))

    if len(pred_id_list) == 1:
        return [0, 0, 0], [0,0, 0, 0]
    true_masks = [None,]
    for t in true_id_list[1:]:
        t_mask = np.array(true == t, np.uint8)
        true_masks.append(t_mask)

    pred_masks = [None,]
    for p in pred_id_list[1:]:
        p_mask = np.array(pred == p, np.uint8)
        pred_masks.append(p_mask)
    pairwise_iou = np.zeros(
        [len(true_id_list) - 1, len(pred_id_list) - 1], dtype=np.float64
    )
    for true_id in true_id_list[1:]:  
        t_mask = true_masks[true_id]
        pred_true_overlap = pred[t_mask > 0]
        pred_true_overlap_id = np.unique(pred_true_overlap)
        pred_true_overlap_id = list(pred_true_overlap_id)
        for pred_id in pred_true_overlap_id:
            if pred_id == 0:  
                continue  
            p_mask = pred_masks[pred_id]
            total = (t_mask + p_mask).sum()
            inter = (t_mask * p_mask).sum()
            iou = inter / (total - inter)
            pairwise_iou[true_id - 1, pred_id - 1] = iou
    #
    if match_iou >= 0.5:
        paired_iou = pairwise_iou[pairwise_iou > match_iou]
        pairwise_iou[pairwise_iou <= match_iou] = 0.0
        paired_true, paired_pred = np.nonzero(pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true += 1  
        paired_pred += 1 
    else:
        paired_true, paired_pred = linear_sum_assignment(-pairwise_iou)
        paired_iou = pairwise_iou[paired_true, paired_pred]
        paired_true = list(paired_true[paired_iou > match_iou] + 1)
        paired_pred = list(paired_pred[paired_iou > match_iou] + 1)
        paired_iou = paired_iou[paired_iou > match_iou]

    unpaired_true = [idx for idx in true_id_list[1:] if idx not in paired_true]
    unpaired_pred = [idx for idx in pred_id_list[1:] if idx not in paired_pred]

    tp = len(paired_true)
    fp = len(unpaired_pred)
    fn = len(unpaired_true)
    dq = tp / (tp + 0.5 * fp + 0.5 * fn)
    sq = paired_iou.sum() / (tp + 1.0e-6)
    pq = dq * sq

    return [dq, sq, pq], [paired_true, paired_pred, unpaired_true, unpaired_pred]