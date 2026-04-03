import torch
import numpy as np
import cv2

from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from training.utils.logger import Log


degree = 1
poly_features = PolynomialFeatures(degree=degree, include_bias=False)
ransac = RANSACRegressor(max_trials=1000)
model = make_pipeline(poly_features, ransac)

def recover_metric_depth_lowres(pred_dpt, batch, b, mask0=None):
    mask0 = (batch['conf'][b][0].detach().cpu().numpy() == 2)
    gt = batch['lowres_dpt'][b][0].detach().cpu().numpy()
    
    h, w = gt.shape[:2]
    pred = cv2.resize(pred_dpt, (w, h), interpolation=cv2.INTER_NEAREST)
    
    
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask = (gt > 1e-8) & (pred > 1e-8)
    mask0 = mask0.squeeze()
    mask0 = mask0 > 0
    mask = mask & mask0
    
    gt_mask = gt[mask]
    pred_mask = pred[mask]
    try:
        # ignore warning 
        a, b = np.polyfit(pred_mask, gt_mask, deg=1)
    except:
        a, b = 1, 0
    if a > 0:
        pred_metric = a * pred_dpt + b
    else:
        pred_mean = np.mean(pred_mask)
        gt_mean = np.mean(gt_mask)
        pred_metric = pred_dpt * (gt_mean / pred_mean)
    return pred_metric

def recover_metric_depth_lowres_ransac(pred_dpt, batch, b, mask0=None):
    mask0 = (batch['conf'][b][0].detach().cpu().numpy() == 2)
    gt = batch['lowres_dpt'][b][0].detach().cpu().numpy()
    
    h, w = gt.shape[:2]
    pred = cv2.resize(pred_dpt, (w, h), interpolation=cv2.INTER_NEAREST)
    
    
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask = (gt > 1e-8) & (pred > 1e-8)
    mask0 = mask0.squeeze()
    mask0 = mask0 > 0
    mask = mask & mask0
    
    gt_mask = gt[mask]
    pred_mask = pred[mask]
    
    try:
        # model.fit(pred_mask[:, None], gt)
        model.fit(pred_mask[:, None], gt_mask[:, None])
        a, b = model.named_steps['ransacregressor'].estimator_.coef_, model.named_steps['ransacregressor'].estimator_.intercept_
        a = a.item()
        b = b.item()
    except:
        a, b = 1, 0
    Log.debug('ransac fit a: {}, b: {}'.format(a, b))
    if a > 0:
        pred_metric = a * pred_dpt + b
    else:
        pred_mean = np.mean(pred_mask)
        gt_mean = np.mean(gt_mask)
        pred_metric = pred_dpt * (gt_mean / pred_mean)
    return pred_metric


def recover_metric_depth(pred, gt, mask0=None, disp=False):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask = (gt > 1e-8) #& (pred > 1e-8)
    if mask0 is not None and mask0.sum() > 0:
        if type(mask0).__module__ == torch.__name__:
            mask0 = mask0.cpu().numpy()
        mask0 = mask0.squeeze()
        mask0 = mask0 > 0
        mask = mask & mask0
    gt_mask = gt[mask]
    pred_mask = pred[mask]
    if disp: gt_mask = 1 / np.clip(gt_mask, 1e-8, None)
    try:
        # ignore warning 
        a, b = np.polyfit(pred_mask, gt_mask, deg=1)
    except:
        import ipdb; ipdb.set_trace()
        a, b = 1, 0
    if a > 0:
        pred_metric = a * pred + b
    else:
        pred_mean = np.mean(pred_mask)
        gt_mean = np.mean(gt_mask)
        pred_metric = pred * (gt_mean / pred_mean)
    if disp: pred_metric = 1 / np.clip(pred_metric, 1e-2, None)
    return pred_metric


def recover_metric_depth_ransac(pred, gt, mask0=None, disp=False, log=False):
    pred = pred.astype(np.float32)
    gt = gt.astype(np.float32)
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = gt.squeeze()
    pred = pred.squeeze()
    mask = (gt > 1e-8) #& (pred > 1e-8)
    if mask0 is not None and mask0.sum() > 0:
        if type(mask0).__module__ == torch.__name__:
            mask0 = mask0.cpu().numpy()
        mask0 = mask0.squeeze()
        mask0 = mask0 > 0
        mask = mask & mask0
    gt_mask = gt[mask].astype(np.float32)
    pred_mask = pred[mask].astype(np.float32)


    if disp: 
        gt_mask = np.clip(gt_mask, 1e-8, None)
        gt_mask = 1 / gt_mask

    if log:
        gt_mask = np.log(gt_mask+1.)

    try:
        model.fit(pred_mask[:, None], gt_mask[:, None])
        a, b = model.named_steps['ransacregressor'].estimator_.coef_, model.named_steps['ransacregressor'].estimator_.intercept_
        a = a.item()
        b = b.item()
    except:
        a, b = 1, 0
        
    if a > 0:
        pred_metric = a * pred + b
    else:
        pred_mean = np.mean(pred_mask)
        gt_mean = np.mean(gt_mask)
        pred_metric = pred * (gt_mean / pred_mean)
    if disp: pred_metric = 1 / np.clip(pred_metric, 1e-2, None)
    if log: pred_metric = np.exp(pred_metric) - 1.
    return pred_metric


def validate_rel_depth_err(pred, gt, smoothed_criteria, mask=None, scale=10.):
    if type(pred).__module__ == torch.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ == torch.__name__:
        gt = gt.cpu().numpy()
    gt = np.squeeze(gt)
    pred = np.squeeze(pred)
    if mask is not None:
        gt = gt[mask[0]:mask[1], mask[2]:mask[3]]
        pred = pred[mask[0]:mask[1], mask[2]:mask[3]]
    if pred.shape != gt.shape:
        import ipdb; ipdb.set_trace()
        return -1
    mask2 = gt > 0
    gt = gt[mask2]
    pred = pred[mask2]

    # invalid evaluation image
    if gt.size < 10:
        return smoothed_criteria

    # Scale matching
    #pred = recover_metric_depth(pred, gt)

    n_pxl = gt.size
    gt_scale = gt * scale
    pred_scale = pred * scale

    # Mean Absolute Relative Error
    rel = np.abs(gt_scale - pred_scale) / gt_scale  # compute errors
    abs_rel_sum = np.sum(rel)
    smoothed_criteria['err_absRel'].AddValue(np.float64(abs_rel_sum), n_pxl)

    # WHDR error
    whdr_err_sum, eval_num = weighted_human_disagreement_rate(gt_scale, pred_scale)
    smoothed_criteria['err_whdr'].AddValue(np.float64(whdr_err_sum), eval_num)
    return smoothed_criteria



def evaluate_rel_err(pred, gt, mask_invalid=None, scale=10.0 ):
    metrics_dict = {}
    if type(pred).__module__ != np.__name__:
        pred = pred.cpu().numpy()
    if type(gt).__module__ != np.__name__:
        gt = gt.cpu().numpy()

    pred = np.squeeze(pred)
    gt = np.squeeze(gt)
    if pred.shape != gt.shape:
        import ipdb; ipdb.set_trace()
        return -1
    if mask_invalid is not None:
        gt = gt[~mask_invalid]
        pred = pred[~mask_invalid]

    mask = (gt > 1e-9) & (pred > 1e-9)
    gt = gt[mask]
    pred = pred[mask]
    n_pxl = gt.size
    gt_scale = gt * scale
    pred_scale = pred * scale

    # invalid evaluation image
    if gt_scale.size < 10:
        print('Valid pixel size:', gt_scale.size, 'Invalid evaluation!!!!')
        return metrics_dict

    #Mean Absolute Relative Error
    rel = np.abs(gt - pred) / gt# compute errors
    abs_rel_sum = np.sum(rel)
    # metrics_dict['err_absRel'] = (np.float64(abs_rel_sum), n_pxl)
    metrics_dict['err_absRel'] = np.float64(abs_rel_sum/n_pxl)

    #Square Mean Relative Error
    s_rel = ((gt_scale - pred_scale) * (gt_scale - pred_scale)) / (gt_scale * gt_scale)# compute errors
    squa_rel_sum = np.sum(s_rel)
    # metrics_dict['err_squaRel'] = (np.float64(squa_rel_sum), n_pxl)
    metrics_dict['err_squaRel'] = np.float64(squa_rel_sum/n_pxl)

    #Root Mean Square error
    square = (gt_scale - pred_scale) ** 2
    rms_squa_sum = np.sum(square)
    # smoothed_criteria['err_rms'].AddValue(np.float64(rms_squa_sum), n_pxl)
    metrics_dict['err_rms'] = np.float64(rms_squa_sum)/n_pxl

    #Log Root Mean Square error
    log_square = (np.log(gt_scale) - np.log(pred_scale)) **2
    log_rms_sum = np.sum(log_square)
    metrics_dict['err_logRms'] = (np.float64(log_rms_sum)/ n_pxl)

    # Scale invariant error
    diff_log = np.log(pred_scale) - np.log(gt_scale)
    diff_log_sum = np.sum(diff_log)
    # smoothed_criteria['err_silog'].AddValue(np.float64(diff_log_sum), n_pxl)
    metrics_dict['err_silog'] = (np.float64(diff_log_sum)/ n_pxl)
    diff_log_2 = diff_log ** 2
    diff_log_2_sum = np.sum(diff_log_2)
    # smoothed_criteria['err_silog2'].AddValue(np.float64(diff_log_2_sum), n_pxl)
    metrics_dict['err_silog2'] = (np.float64(diff_log_2_sum)/ n_pxl)

    # Mean log10 error
    log10_sum = np.sum(np.abs(np.log10(gt) - np.log10(pred)))
    metrics_dict['err_log10'] = (np.float64(log10_sum)/ n_pxl)

    #Delta
    gt_pred = gt_scale / pred_scale
    pred_gt = pred_scale / gt_scale
    gt_pred = np.reshape(gt_pred, (1, -1))
    pred_gt = np.reshape(pred_gt, (1, -1))
    gt_pred_gt = np.concatenate((gt_pred, pred_gt), axis=0)
    ratio_max = np.amax(gt_pred_gt, axis=0)

    delta_1_sum = np.sum(ratio_max < 1.25)
    metrics_dict['err_delta1'] = (np.float64(delta_1_sum)/ n_pxl)
    delta_2_sum = np.sum(ratio_max < 1.25**2)
    metrics_dict['err_delta2'] = (np.float64(delta_2_sum)/ n_pxl)
    delta_3_sum = np.sum(ratio_max < 1.25**3)
    metrics_dict['err_delta3'] = (np.float64(delta_3_sum)/ n_pxl)

    # WHDR error
    whdr_err_sum, eval_num = weighted_human_disagreement_rate(gt_scale, pred_scale)
    metrics_dict['err_whdr'] = (np.float64(whdr_err_sum)/ eval_num)
    
    # L1 
    metrics_dict['dpt_l1'] = np.float64(np.mean(np.abs(gt - pred)))
    metrics_dict['dpt_rmse'] = np.float64(np.sqrt(np.mean((gt - pred) ** 2)))

    return metrics_dict


def weighted_human_disagreement_rate(gt, pred):
    p12_index = select_index(gt)
    gt_reshape = np.reshape(gt, gt.size)
    pred_reshape = np.reshape(pred, pred.size)
    mask = gt > 0
    gt_p1 = gt_reshape[mask][p12_index['p1']]
    gt_p2 = gt_reshape[mask][p12_index['p2']]
    pred_p1 = pred_reshape[mask][p12_index['p1']]
    pred_p2 = pred_reshape[mask][p12_index['p2']]

    p12_rank_gt = np.zeros_like(gt_p1)
    p12_rank_gt[gt_p1 > gt_p2] = 1
    p12_rank_gt[gt_p1 < gt_p2] = -1

    p12_rank_pred = np.zeros_like(gt_p1)
    p12_rank_pred[pred_p1 > pred_p2] = 1
    p12_rank_pred[pred_p1 < pred_p2] = -1

    err = np.sum(p12_rank_gt != p12_rank_pred)
    valid_pixels = gt_p1.size
    return err, valid_pixels


def select_index(gt_depth, select_size=10000):
    valid_size = np.sum(gt_depth>0)
    try:
        p = np.random.choice(valid_size, select_size*2, replace=False)
    except:
        p = np.random.choice(valid_size, select_size*2*2, replace=True)
    np.random.shuffle(p)
    p1 = p[0:select_size*2:2]
    p2 = p[1:select_size*2:2]

    p12_index = {'p1': p1, 'p2': p2}
    return p12_index