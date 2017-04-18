import numpy as np
import math
import cv2
import os
#from scipy.special import expit
#from utils.box import BoundBox, box_iou, prob_compare
#from utils.box import prob_compare2, box_intersection
from utils.box import BoundBox
from cython_utils.cy_yolo2_findboxes import box_constructor

def expit(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x):
    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out

def findboxes(self, net_out):
    # meta
    meta = self.meta
    boxes = list()
    boxes=box_constructor(meta,net_out)
    return boxes

def add_current_box(self, im, box):
    def _add_box(imgcv, box):
        left, top, right, bot = box["box"]
        color = box["color"]
        thick = box["thick"]
        label = box["label"]
        cv2.rectangle(imgcv,
            (left, top), (right, bot),
            color, thick)
        cv2.putText(imgcv, label, (left, top - 12),
            0, 1e-3 * h, color,thick//3)

    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else: imgcv = im
    h, w, _ = imgcv.shape
    for b in box:
        _add_box(imgcv, b)
    return imgcv

def postprocess(self, net_out, im, save = True, return_pred=True):
    """
    Takes net output, draw net_out, save to disk
    """
    boxes = self.findboxes(net_out)

    # meta
    meta = self.meta
    threshold = meta['thresh']
    colors = meta['colors']
    labels = meta['labels']
    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else: imgcv = im
    h, w, _ = imgcv.shape
    
    textBuff = "["
    pred = []
    current_boxes = []
    for b in boxes:
        boxResults = self.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, mess, max_indx, confidence = boxResults
        thick = int((h + w) // 300)
        pred.append({
            "label": mess,
            "confidence": confidence,
            "box": [left, right, top, bot],
        })
        if self.FLAGS.json:
            line =     ('{"label":"%s",'
                    '"confidence":%.2f,'
                    '"topleft":{"x":%d,"y":%d},'
                    '"bottomright":{"x":%d,"y":%d}},\n') % \
                    (mess, confidence, left, top, right, bot)
            textBuff += line
            continue

        cv2.rectangle(imgcv,
            (left, top), (right, bot),
            colors[max_indx], thick)
        mess = "{}:{:.2f}".format(mess, confidence)
        cv2.putText(imgcv, mess, (left, top - 12),
            0, 1e-3 * h, colors[max_indx],thick//3)
        current_boxes.append({
            "box": [left, top, right, bot],
            "color": colors[max_indx],
            "thick": thick,
            "label": mess,
        })

    if not save: return imgcv, pred, current_boxes

def postprocess_for_api(self, net_out, im, save=False):
    boxes = self.findboxes(net_out)

    # meta
    meta = self.meta
    threshold = meta['thresh']
    colors = meta['colors']
    labels = meta['labels']
    if type(im) is not np.ndarray:
        imgcv = cv2.imread(im)
    else: imgcv = im
    h, w, _ = imgcv.shape
    
    textBuff = "["
    pred = []
    current_boxes = []
    for b in boxes:
        boxResults = self.process_box(b, h, w, threshold)
        if boxResults is None:
            continue
        left, right, top, bot, mess, max_indx, confidence = boxResults
        thick = int((h + w) // 300)
        pred.append({
            "label": mess,
            "confidence": float(confidence),
            "box": [left, right, top, bot],
        })
        current_boxes.append({
            "box": [left, top, right, bot],
            "color": colors[max_indx],
            "thick": thick,
            "label": mess,
        })
        if save:
            cv2.rectangle(imgcv,
                (left, top), (right, bot),
                colors[max_indx], thick)
            mess = "{}:{:.2f}".format(mess, confidence)
            cv2.putText(imgcv, mess, (left, top - 12),
                0, 1e-3 * h, colors[max_indx],thick//3)
    pred.sort(key=lambda x:x["confidence"], reverse=True)
    return imgcv, pred, current_boxes
