from net.build import TFNet
import cv2
import pprint
import sys, os

input_video = sys.argv[1]
if not os.path.exists(input_video):
    print "{} doesn't exist!".format(input_video)
    sys.exit(0)
out_video = "{}_detect.mp4".format(input_video.split(".")[0])
if os.path.exists(out_video):
    os.remove(out_video)

options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "tiny-yolo-voc.weights", "threshold": 0.1, "gpu": 1.0, "skip": 10}

tfnet = TFNet(options)
preds = tfnet.camera(input_video, out_video, True)
#preds = tfnet.camera("faizon.mp4", "out.avi", True)
pprint.pprint(preds)
