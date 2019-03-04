#!/usr/bin/env python


# rospy for the subscriber
import rospy
import rospkg
# ROS Image message
from sensor_msgs.msg import Image
import cv2
import torch

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo
from maskrcnn_benchmark.utils import cv2_util
from threading import Thread

import time
from cv_bridge import CvBridge, CvBridgeError


def compute_colors_for_labels( labels):
    """
    Simple function that adds fixed colors depending on the class
    """
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")
    return colors

def overlay_mask(image, predictions):
        """
        Adds the instances contours for each predicted object.
        Each label has a different color.

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `mask` and `labels`.
        """
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = compute_colors_for_labels(labels).tolist()

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None]
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 3)

        composite = image

        return composite


image_processing = False
top_predictions = None #top predictions of maskrcnn
def image_callback(img,args):
    global image_processing
    global top_predictions
    try:
        if image_processing ==False:
            image_processing = True  
            pub = args[0]
            coco_demo = args[1]
            #img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
            rospy.loginfo("recieved")
            top_predictions = coco_demo.run_on_opencv_image(img)
            # img = overlay_mask(img, top_predictions)
            # cv2.imshow("processing", composite)
            # cv2.waitKey(5000)
            # cv2.destroyAllWindows()
            # msg_frame = CvBridge().cv2_to_imgmsg(composite)
            # pub.publish(msg_frame)
            image_processing = False
    except:
        image_processing = False



def display_image(msg,args):
    img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
    if image_processing ==False:
        thread = Thread(target = image_callback, args =(img,args))
        thread.start()
    
    if top_predictions != None:
        img = overlay_mask(img, top_predictions)
    rospy.loginfo("recieved")
    #composite = coco_demo.run_on_opencv_image(img)
    cv2.imshow("processing", img)
    cv2.waitKey(1)



def main():
    #get the root of the packacge
    rospack = rospkg.RosPack()
    package_root = rospack.get_path('maskrcnnpkg')

    rospy.init_node('image_editor')
    rospy.loginfo("in main")
    # Define your image topic

    #create the setup the maskrcnn
    cfg.merge_from_file(package_root+"/scripts/pre_trained_models/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml")
    cfg.merge_from_list(['MODEL.DEVICE', 'cpu'])
    cfg.freeze()

    coco_demo = COCODemo(
        cfg,
        confidence_threshold=0.7,
        masks_per_dim=2,
        min_image_size=250,
    )
    # cam = cv2.VideoCapture(1)
    # ret_val, img = cam.read()
    # composite = coco_demo.run_on_opencv_image(img)
    # cv2.imshow("processing", composite)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    image_topic = "/camera_end_effector/image_raw"
    edited_topic = "/image_processed"
    # Set up your subscriber and define its callback
    pub = rospy.Publisher(edited_topic, Image, queue_size=10)
    sub = rospy.Subscriber(image_topic, Image, display_image,callback_args = [pub,coco_demo])

    #pub.publish(msg_frame)
    #rospy.Subscriber(stream_image_topic, Image, image_callback, callback_args=pub)
    # Spin until ctrl + c
    rospy.loginfo("subscribed")
    rospy.spin()

if __name__ == '__main__':
    main()
