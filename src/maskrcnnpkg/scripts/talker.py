#! /usr/bin/python
# Copyright (c) 2016 Rethink Robotics, Inc.

# rospy for the subscriber
import rospy
import rospkg
import skimage.io
import cv2
# ROS Image message
from sensor_msgs.msg import Image

from cv_bridge import CvBridge, CvBridgeError

def image_callback(msg, pub):
    pub.publish(msg)

def main():
     #get the root of the packacge
    rospack = rospkg.RosPack()
    package_root = rospack.get_path('maskrcnnpkg')


    rospy.init_node('image_publisher')
    rospy.loginfo("in main")
    # Define your image topic
    image = cv2.imread(package_root+'/scripts/test.jpg')
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #   
    msg_frame = CvBridge().cv2_to_imgmsg(image)
    # Define the Display topic
    xdisplay_topic = "/image_topic"
    # Set up your subscriber and define its callback
    pub = rospy.Publisher(xdisplay_topic, Image, queue_size=10)
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        pub.publish(msg_frame)
        rospy.loginfo("Published image")
        rate.sleep()
    #rospy.Subscriber(stream_image_topic, Image, image_callback, callback_args=pub)
    # Spin until ctrl + c
    rospy.loginfo("published")
    rospy.spin()

if __name__ == '__main__':
    main()


