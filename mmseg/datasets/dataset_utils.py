import numpy as np

def per_class_metrics_from_conf(conf):
    # 每类 precision/recall/IoU
    tp = np.diag(conf).astype(float)
    pred_sum = conf.sum(0).astype(float)  # 列和（被预测为该类）
    gt_sum   = conf.sum(1).astype(float)  # 行和（真实属于该类）
    union    = gt_sum + pred_sum - tp

    prec = np.divide(tp, pred_sum, out=np.zeros_like(tp), where=pred_sum>0)
    rec  = np.divide(tp, gt_sum,   out=np.zeros_like(tp), where=gt_sum>0)
    iou  = np.divide(tp, union,    out=np.zeros_like(tp), where=union>0)
    f1   = np.divide(2*prec*rec, (prec+rec), out=np.zeros_like(tp), where=(prec+rec)>0)
    return prec, rec, f1, iou

def macro_averages(prec, rec, f1, iou):
    mprec   = np.nanmean(prec)
    mrecall = np.nanmean(rec)
    mf1     = np.nanmean(f1)      # 宏F1建议按“每类F1再平均”
    miou    = np.nanmean(iou)
    return mprec, mrecall, mf1, miou

def freq_weighted_iou(conf):
    freq = conf.sum(1)
    tp = np.diag(conf).astype(float)
    pred_sum = conf.sum(0).astype(float)
    gt_sum   = conf.sum(1).astype(float)
    union    = gt_sum + pred_sum - tp
    iou = np.divide(tp, union, out=np.zeros_like(tp), where=union>0)
    fw_iou = np.sum(freq * iou) / np.sum(freq)
    return fw_iou

def getScores_self(conf_matrix):
    # 使用
    prec, rec, f1, iou = per_class_metrics_from_conf(conf_matrix)
    # 正类（假设索引1是road）
    prec_road, rec_road, f1_road, iou_road = prec[1], rec[1], f1[1], iou[1]
    # 宏/加权
    mprec, mrecall, mf1, miou = macro_averages(prec, rec, f1, iou)
    fwIoU = freq_weighted_iou(conf_matrix)

    return mprec, mrecall, mf1, miou, fwIoU, prec_road, rec_road, f1_road, iou_road