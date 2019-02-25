#!/usr/bin/env python


# rospy for the subscriber
import rospy
import rospkg
# ROS Image message
from sensor_msgs.msg import Image
import cv2


from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

from cv_bridge import CvBridge, CvBridgeError

image_processing = False
def image_callback(msg,args):
    global image_processing
    try:
        if image_processing ==False:
            image_processing = True  
            pub = args[0]
            coco_demo = args[1]
            img = CvBridge().imgmsg_to_cv2(msg, desired_encoding="passthrough")
            rospy.loginfo("recieved")
            composite = coco_demo.run_on_opencv_image(img)
            # cv2.imshow("processing", composite)
            # cv2.waitKey(5000)
            # cv2.destroyAllWindows()
            msg_frame = CvBridge().cv2_to_imgmsg(composite)
            pub.publish(msg_frame)
            image_processing = False
    except:
        image_processing = False
    


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
    image_topic = "/image_topic"
    edited_topic = "/image_processed"
    # Set up your subscriber and define its callback
    pub = rospy.Publisher(edited_topic, Image, queue_size=10)
    sub = rospy.Subscriber(image_topic, Image, image_callback,callback_args = [pub,coco_demo])

    #pub.publish(msg_frame)
    #rospy.Subscriber(stream_image_topic, Image, image_callback, callback_args=pub)
    # Spin until ctrl + c
    rospy.loginfo("subscribed")
    rospy.spin()

if __name__ == '__main__':
    main()
