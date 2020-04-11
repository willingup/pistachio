
#include <cv_bridge/cv_bridge.h>
#include <cstdlib>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/opencv.hpp"
#include "ros/ros.h"
#include "ros/console.h"
#include "linedetect.hpp"
#include "xr_line_follower/pos.h"

/**
 *@brief Main function that reads image from the turtlebot and provides direction to move using image processing
 *@param argc is the number of arguments of the main function
 *@param argv is the array of arugments
 * linedetect.cpp中写了3个函数,mian调用它们实现图像处理,其实detect.cpp和linedetect.cpp可以合并
 *@return 0
*/

int main(int argc, char **argv) {
    // Initializing node and object 初始化节点和对象
    ros::init(argc, argv, "detection");
    ros::NodeHandle n;
    LineDetect det;

    // Creating Publisher and subscriber 创建发布着和订阅
    ros::Subscriber sub = n.subscribe("/camera/rgb/image_raw",
        1, &LineDetect::imageCallback, &det);
    
    //消息发布 （消息类型 消息名 缓冲消息条数上限（队列缓冲机制，丢弃旧消息））
    ros::Publisher dirPub = n.advertise<xr_line_follower::pos>("direction", 1);
    xr_line_follower::pos msg;

    while (ros::ok()) {
        if (!det.img.empty()) {
            // Perform image processing 执行图像处理
            det.img_filt = det.Gauss(det.img);
            msg.direction = det.colorthresh(det.img_filt);
            // Publish direction message 将方向信息发布
            dirPub.publish(msg);
            }
        ros::spinOnce();
    }
    // Closing image viewer
    cv::destroyWindow("XRrobot View");
}
