import numpy as np
from scipy import interpolate


def evaluate_deep_lesion(results, dataset, cfg, logger):
    pred_bboxes = []
    gt_bboxes = []
    
    for idx in range(len(dataset)):
        ann_info = dataset.get_ann_info(idx)
        gt_bboxes.append(ann_info['bboxes'])
        pred_bboxes.append(results[idx][0])  # only for fg
    logger.info('Sensitivity @{} average FPs per image:'.format(cfg.val_avg_fp))

    res = sens_at_FP(pred_bboxes, gt_bboxes, cfg.val_avg_fp, cfg.val_iou_th)  # cls 0 is background
    logger.info(res)
    return res[3]  # sens@4FP



def sens_at_FP(pred_bboxes, gt_bboxes, avgFP, iou_th):
    # compute the sensitivity at avgFP (average FP per image)
    sens, fp_per_img = FROC(pred_bboxes, gt_bboxes, iou_th)
    f = interpolate.interp1d(fp_per_img, sens, fill_value='extrapolate')
    res = f(np.array(avgFP))
    return res


def FROC(pred_bboxes, gt_bboxes, iou_th):
    # Compute the FROC curve, for single class only
    nImg = len(pred_bboxes)
    img_idxs = np.hstack([[i] * len(pred_bboxes[i]) for i in range(nImg)]).astype('int')
    boxes_cat = np.vstack(pred_bboxes)
    scores = boxes_cat[:, -1]
    ord = np.argsort(scores)[::-1]
    boxes_cat = boxes_cat[ord, :4]
    img_idxs = img_idxs[ord]

    hits = [np.zeros((len(gts),), dtype=bool) for gts in gt_bboxes]
    nHits = 0
    nMiss = 0
    tps = []
    fps = []
    for i in range(len(boxes_cat)):
        overlaps = IOU(boxes_cat[i, :], gt_bboxes[img_idxs[i]])
        if overlaps.max() < iou_th:
            nMiss += 1
        else:
            for j in range(len(overlaps)):
                if overlaps[j] >= iou_th and not hits[img_idxs[i]][j]:
                    hits[img_idxs[i]][j] = True
                    nHits += 1

        tps.append(nHits)
        fps.append(nMiss)

    nGt = len(np.vstack(gt_bboxes))
    sens = np.array(tps, dtype=float) / nGt
    fp_per_img = np.array(fps, dtype=float) / nImg

    return sens, fp_per_img


def IOU(box1, gts):
    # compute overlaps
    # intersection
    ixmin = np.maximum(gts[:, 0], box1[0])
    iymin = np.maximum(gts[:, 1], box1[1])
    ixmax = np.minimum(gts[:, 2], box1[2])
    iymax = np.minimum(gts[:, 3], box1[3])
    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)
    inters = iw * ih

    # union
    uni = ((box1[2] - box1[0] + 1.) * (box1[3] - box1[1] + 1.) +
           (gts[:, 2] - gts[:, 0] + 1.) *
           (gts[:, 3] - gts[:, 1] + 1.) - inters)

    overlaps = inters / uni
    # ovmax = np.max(overlaps)
    # jmax = np.argmax(overlaps)
    return overlaps


def num_true_positive(boxes, gts, num_box, iou_th):
    # only count once if one gt is hit multiple times
    hit = np.zeros((gts.shape[0],), dtype=np.bool)
    scores = boxes[:, -1]
    boxes = boxes[scores.argsort()[::-1], :4]

    for i, box1 in enumerate(boxes):
        if i == num_box: break
        overlaps = IOU(box1, gts)
        hit = np.logical_or(hit, overlaps >= iou_th)

    tp = np.count_nonzero(hit)

    return tp


def recall_all(pred_bboxes, gt_bboxes, num_box, iou_th):
    # Compute the recall at num_box candidates per image
    nCls = len(pred_bboxes)
    nImg = len(pred_bboxes[0])
    recs = np.zeros((nCls, len(num_box)))
    nGt = np.zeros((nCls,), dtype=np.float)

    for cls in range(nCls):
        for i in range(nImg):
            nGt[cls] += gt_bboxes[cls][i].shape[0]
            for n in range(len(num_box)):
                tp = num_true_positive(pred_bboxes[cls][i], gt_bboxes[cls][i], num_box[n], iou_th)
                recs[cls, n] += tp

    recs /= nGt
    return recs
