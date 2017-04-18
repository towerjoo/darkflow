from net.build import TFNet
import cv2
import pprint
import sys, os

input_video = sys.argv[1]
if not os.path.exists(input_video):
    print "{} doesn't exist!".format(input_video)
    sys.exit(0)

options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "tiny-yolo-voc.weights", "threshold": 0.1, "gpu": 1.0, "window": 10}

tfnet = TFNet(options)
preds = tfnet.analyze_video(input_video)
#preds = tfnet.camera("faizon.mp4", "out.avi", True)
pprint.pprint(preds)
