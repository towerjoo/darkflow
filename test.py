from net.build import TFNet
import cv2
import pprint

options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "tiny-yolo-voc.weights", "threshold": 0.1, "gpu": 1.0}

tfnet = TFNet(options)
preds = tfnet.camera("test.mp4", "out.avi", True)
pprint.pprint(preds)
