/*
 * ros_caffe_test.cpp
 *
 * 
 */

#include <ros/ros.h>
#include <ros/package.h>
#include <std_msgs/String.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include "Classifier.h"

const std::string RECEIVE_IMG_TOPIC_NAME = "camera/rgb/image_raw";
const std::string PUBLISH_RET_TOPIC_NAME = "/caffe_ret";

Classifier* classifier;

//caffemodel相关的4个文件
std::string model_path;
std::string weights_path;
std::string mean_file;
std::string label_file;

//被检测图片的位置
std::string image_path;

ros::Publisher gPublisher;

//publishRet函数的声明
void publishRet(const std::vector<Prediction>& predictions);

//回调函数 话题中的图像到达时，会自动的调用这个函数
void imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
	//sensor_msgs数据转成opencv数据类型
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
	
        //cv::imwrite("rgb.png", cv_ptr->image);
		cv::Mat img = cv_ptr->image;
		
		std::vector<Prediction> predictions = classifier->Classify(img);
		publishRet(predictions);
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

// TODO: Define a msg or create a service
// Try to receive : $rostopic echo /caffe_ret
void publishRet(const std::vector<Prediction>& predictions)  {
    std_msgs::String msg;
    std::stringstream ss;
    for (size_t i = 0; i < predictions.size(); ++i) {
        Prediction p = predictions[i];
        ss << "[" << p.second << " - " << p.first << "]" << std::endl;
    }
    msg.data = ss.str();
    gPublisher.publish(msg);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "ros_caffe_test");
    
    //创建Nodehandle句柄，是ros节点的强制句柄
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    
    // To receive an image from the topic, PUBLISH_RET_TOPIC_NAME
    image_transport::Subscriber sub = it.subscribe(RECEIVE_IMG_TOPIC_NAME, 1, imageCallback);
    
    //消息发布 （消息类型 消息名 缓冲消息条数上限（队列缓冲机制，丢弃旧消息））
    gPublisher = nh.advertise<std_msgs::String>(PUBLISH_RET_TOPIC_NAME, 100);
    
    //与模型相关的文件
    const std::string ROOT_SAMPLE = ros::package::getPath("ros_caffe");
    model_path = ROOT_SAMPLE + "/data/deploy.prototxt";
    weights_path = ROOT_SAMPLE + "/data/bvlc_reference_caffenet.caffemodel";
    mean_file = ROOT_SAMPLE + "/data/imagenet_mean.binaryproto";
    label_file = ROOT_SAMPLE + "/data/synset_words.txt";
    image_path = ROOT_SAMPLE + "/data/cat.jpg";

    classifier = new Classifier(model_path, weights_path, mean_file, label_file);
    
    // Test data/cat.jpg 这是测一张图片
    cv::Mat img = cv::imread(image_path, -1);
    std::vector<Prediction> predictions = classifier->Classify(img);
    
    /* Print the top N predictions. */
    std::cout << "Test default image under /data/cat.jpg" << std::endl;
    for (size_t i = 0; i < predictions.size(); ++i) {
        Prediction p = predictions[i];
        std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
    }
	publishRet(predictions);
    
    ros::spin();
    delete classifier;
    ros::shutdown();
    return 0;
}
