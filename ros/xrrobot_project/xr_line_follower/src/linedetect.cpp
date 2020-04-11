#include "linedetect.hpp"
#include <cv_bridge/cv_bridge.h>
#include <cstdlib>
#include <string>
#include <opencv2/highgui/highgui.hpp>
#include "ros/ros.h"
#include "opencv2/opencv.hpp"
#include "ros/console.h"
#include "xr_line_follower/pos.h"

//回调用于从Turtlebot订阅图像主题并将其转换为opencv图像格式
void LineDetect::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
  cv_bridge::CvImagePtr cv_ptr;
  try {
    cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    img = cv_ptr->image;
    cv::waitKey(30);
  }
  catch (cv_bridge::Exception& e) {
    ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
  }
}



cv::Mat LineDetect::Gauss(cv::Mat input) {
  cv::Mat output;
  //Applying Gaussian Filter 高斯滤波
  //输入图像和输出图像，Ksize为高斯滤波器模板大小,sigmaX和sigmaY分别为高斯滤波在横线和竖向的滤波系数,borderType为边缘点插值类型。
  cv::GaussianBlur(input, output, cv::Size(3, 3), 0.1, 0.1);
  return output;
}


//功能使用颜色阈值，图像蒙版和质心检测执行发布线检测以发布方向，input是opencv矩阵格式的过滤后的输入图像, return int direction返回乌龟应该进入的方向
int LineDetect::colorthresh(cv::Mat input) {
  
  // Initializaing variables
  cv::Size s = input.size();
  std::vector<std::vector<cv::Point> > v;
  auto w = s.width;
  auto h = s.height;
  auto c_x = 0.0;
  
  // Detect all objects within the HSV range 检测HSV范围RGBToHSV内的所有对象
  //色彩空间的转化
  cv::cvtColor(input, LineDetect::img_hsv, CV_BGR2HSV);
  
  //利用颜色范围区分检测的物体
  //LineDetect::LowerBlack = {0,0,0};
  //LineDetect::UpperBlack = {150,255,46};

  LineDetect::LowerYellow = {0, 100, 50};
  LineDetect::UpperYellow = {34, 200, 200};

  //LineDetect::LowerYellow = {20, 100, 100};
  //LineDetect::UpperYellow = {30, 255, 255};
  cv::inRange(LineDetect::img_hsv, LowerYellow,UpperYellow, LineDetect::img_mask);

  //cv::inRange(LineDetect::img_hsv, LowerBlack,
  // UpperBlack, LineDetect::img_mask)
  
  rosrun rqt_reconfigure rqt_reconfigure
  img_mask(cv::Rect(0, 0, w, 0.8*h)) = 0;
  
  // Find contours for better visualization 查找轮廓以获得更好的可视化
  cv::findContours(LineDetect::img_mask, v, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);
  // If contours exist add a bounding	   如果存在轮廓，则添加边界
  // Choosing contours with maximum area   选择最大的轮廓
  if (v.size() != 0) {
  auto area = 0;
  auto idx = 0;
  auto count = 0;
  while (count < v.size()) {
    if (area < v[count].size()) {
       idx = count;
       area = v[count].size();
    }
    count++;
  }
  cv::Rect rect = boundingRect(v[idx]);
  cv::Point pt1, pt2, pt3;
  pt1.x = rect.x;
  pt1.y = rect.y;
  pt2.x = rect.x + rect.width;
  pt2.y = rect.y + rect.height;
  pt3.x = pt1.x+5;
  pt3.y = pt1.y-5;
  
  // Drawing the rectangle using points obtained 使用点描绘矩形
  rectangle(input, pt1, pt2, CV_RGB(255, 0, 0), 2);
  
  // Inserting text box    插入文本框
  cv::putText(input, "Line Detected", pt3,
    CV_FONT_HERSHEY_COMPLEX, 1, CV_RGB(255, 0, 0));
  }
  
  // Mask image to limit the future turns affecting the output  遮罩图像以限制将来的转向影响输出
  img_mask(cv::Rect(0.7*w, 0, 0.3*w, h)) = 0;
  img_mask(cv::Rect(0, 0, 0.3*w, h)) = 0;
  
  // Perform centroid detection of line 执行线的质心检测
  cv::Moments M = cv::moments(LineDetect::img_mask);
  if (M.m00 > 0) {
    cv::Point p1(M.m10/M.m00, M.m01/M.m00);
    cv::circle(LineDetect::img_mask, p1, 5, cv::Scalar(155, 200, 0), -1); //绘制质心
  }
  c_x = M.m10/M.m00;
  // Tolerance to chooise directions  选择方向
  auto tol = 15;
  auto count = cv::countNonZero(img_mask);
  // Turn left if centroid is to the left of the image center minus tolerance
  // Turn right if centroid is to the right of the image center plus tolerance
  // Go straight if centroid is near image center
  
  //如果质心在图像中心的左侧减去公差，则向左转
  //如果质心在图像中心的右侧加上公差，则向右转
  //如果质心在图像中心附近，请直走
  
  if (c_x < w/2-tol) {
    LineDetect::dir = 0;
  } else if (c_x > w/2+tol) {
    LineDetect::dir = 2;
  } else {
    LineDetect::dir = 1;
  }
  // Search if no line detected  搜索是否未检测到线
  if (count == 0) {
    LineDetect::dir = 3;
  }
  // Output images viewed by the turtlebot 输出机器人查看的图像
  cv::namedWindow("XRrobot View");
  imshow("XRrobot View", input);
  return LineDetect::dir;
}
