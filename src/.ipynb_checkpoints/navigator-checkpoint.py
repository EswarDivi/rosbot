#!/usr/bin/env python3

import rospy
import torch
import torchvision
from torch2trt import TRTModule
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import PIL.Image
import numpy as np
import time
import random
from jetcam.usb_camera import USBCamera
from std_msgs.msg import String

print("Imported libraries")
device = torch.device("cuda")
model_trt = TRTModule()
model_trt.load_state_dict(torch.load("/home/nikhil/ROS_project/best_model_resnet18.pth_trt.pth"))
print("Loaded model")
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda().half()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda().half()
normalize = torchvision.transforms.Normalize(mean, std)


def preprocess(image):
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device).half()
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


camera = USBCamera(
    width=224, height=224, capture_width=640, capture_height=480, capture_device=0
)
image = camera.read()
camera.running = True
print("Camera connected! ")



rospy.init_node('navigator', anonymous=True)
pub= rospy.Publisher("direction", String, queue_size=10)
print("Node created")
# def talker ():
#     while not rospy.is_shutdown ():
#         output = "#some code here"
#         rospy.loginfo(output)
#         pub.publish(output)


directions = ["l", "r"]
prev_output = "f"

def update(change):
    global prev_output, directions
    x = change["new"]
    x = preprocess(x)
    y = model_trt(x)
    # print(y)

    # we apply the `softmax` function to normalize the output vector so it sums to 1 (which makes it a probability distribution)
    y = F.softmax(y, dim=1)
    # print(y)

    prob_blocked = float(y.flatten()[0])
    if prob_blocked > 0.5:
        if prev_output is "f":
            sampled = random.choices(directions, k=1)
            sampled = sampled[0]
            prev_output = sampled
    else:
        prev_output = "f"
    pub.publish(prev_output)
    time.sleep(0.001)

update({"new": camera.value})  # we call the function once to initialize

camera.observe(
    update, names="value"
)  # this attaches the 'update' function to the 'value' traitlet of our camera
