代码解析参考于：https://blog.csdn.net/maweifei/article/details/72811413
/*
 * Classifier.h   
 */

#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <iostream>
#include <vector>
#include <sstream>
#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace caffe;
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

class Classifier {
    public:
        /* 
        model_file为测试模型时记录网络结构的prototxt文件
        trained_file为训练完毕的caffemodel文件路径  
        mean_file为记录数据集均值的文件路径，数据集均值的文件
        label_file为记录类别标签的文件路径，标签通常记录在一个txt
         */
        
        Classifier(const string& model_file,
                   const string& trained_file,
                   const string& mean_file,
                   const string& label_file);
        //Classify函数去进行网络前传，得到img属于各个类的概率  
        std::vector<Prediction> Classify(const cv::Mat& img, int N = 5);

    private:
        //SetMean函数主要进行均值设定，每张检测图输入后会进行减去均值的操作，这个均值可以是模型使用的数据集图像的均值 
        void SetMean(const string& mean_file);
        //Predict函数是Classify函数的主要组成部分，将img送入网络进行前向传播，得到最后的类别  
        std::vector<float> Predict(const cv::Mat& img);
        //WrapInputLayer函数将img各通道(input_channels)放入网络的输入blob中  
        void WrapInputLayer(std::vector<cv::Mat>* input_channels);
        //Preprocess函数将输入图像img按通道分开(input_channels)    
        void Preprocess(const cv::Mat& img,
                        std::vector<cv::Mat>* input_channels);

    private:
        /* 
        net_表示caffe中的网络
        input_geometry_表示了输入图像的高宽，同时也是网络数据层中单通道图像的高宽  
        num_channels_表示了输入图像的通道数  
        mean_表示了数据集的均值，格式为Mat  
        字符串向量labels_表示了各个标签
         */
        shared_ptr<Net<float> > net_;
        cv::Size input_geometry_;
        int num_channels_;
        cv::Mat mean_;
        std::vector<string> labels_;
};

 //构造函数Classifier进行了各种各样的初始化工作，并对网络的安全进行了检验  
Classifier::Classifier(const string& model_file,
                       const string& trained_file,
                       const string& mean_file,
                       const string& label_file) {
#ifdef CPU_ONLY
    Caffe::set_mode(Caffe::CPU);
#else
    Caffe::set_mode(Caffe::GPU);
#endif

    /* Load the network. */
    //prototxt初始化网络结构
    net_.reset(new Net<float>(model_file, TEST));
    net_->CopyTrainedLayersFrom(trained_file);      //从trained_file路径下的caffemodel文件读入训练完毕的网络参数  
  

    Blob<float>* input_layer = net_->input_blobs()[0];      //获取网络输入的blob，表示网络的数据层
    num_channels_ = input_layer->channels(); //获取输入的通道数

    input_geometry_ = cv::Size(input_layer->width(), input_layer->height());//获取输入图像的尺寸(宽与高

    /* Load the binaryproto mean file. */
    SetMean(mean_file);   //进行均值的设置

    /* Load labels. */
    std::ifstream labels(label_file.c_str()); //读入定义的标签文件
    string line;  //定义line,用来获取标签文件中的每一行即每一个标签
    //循环将标签读入labels中
    while (std::getline(labels, line))
        labels_.push_back(string(line));
     /*
     output_layer指向网络最后的输出
     举个例子，最后的分类器采用softmax分类，且类别有10类，那么，输出的blob就会有10个通道，每个通道的长 宽都为1
     (因为是10个数，这10个数表征输入属于10类中每一类的概率，这10个数之和应该为1)，输出blob的结构为(1,10,1,1)
  */          
    Blob<float>* output_layer = net_->output_blobs()[0];
}
/*
PairCompare函数比较分类得到的物体属于某两个类别的概率的大小，
若属于lhs的概率大于属于rhs的概率，返回真，否则返回假 
*/
static bool PairCompare(const std::pair<float, int>& lhs,
                        const std::pair<float, int>& rhs) {
    return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
    std::vector<std::pair<float, int> > pairs;
    for (size_t i = 0; i < v.size(); ++i)
        pairs.push_back(std::make_pair(v[i], i));
    std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);

    std::vector<int> result;
    for (int i = 0; i < N; ++i)
        result.push_back(pairs[i].second);
    return result;
}

/* Return the top N predictions. */
std::vector<Prediction> Classifier::Classify(const cv::Mat& img, int N) {
    std::vector<float> output = Predict(img);

    std::vector<int> maxN = Argmax(output, N);
    std::vector<Prediction> predictions;
    for (int i = 0; i < N; ++i) {
        int idx = maxN[i];
        predictions.push_back(std::make_pair(labels_[idx], output[idx]));
    }

    return predictions;
}

/* Load the mean file in binaryproto format. */
void Classifier::SetMean(const string& mean_file) { //设置数据集的平均值
    BlobProto blob_proto;

    //用定义的均值文件路径将均值文件读入proto中  
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);

    /* Convert from BlobProto to Blob<float> */
    Blob<float> mean_blob;
    //将proto中存储的均值文件转移到blob中 
    mean_blob.FromProto(blob_proto);

    /* The format of the mean file is planar 32-bit float BGR or grayscale. */
    //将mean_blob中的数据转化为Mat时的存储向量 
    std::vector<cv::Mat> channels;

    //指向均值blob的指针  
    float* data = mean_blob.mutable_cpu_data();

    for (int i = 0; i < num_channels_; ++i) {
        /* Extract an individual channel. */

        //存储均值文件的每一个通道转化得到的Mat  
        cv::Mat channel(mean_blob.height(), mean_blob.width(), CV_32FC1, data);
        //将均值文件的所有通道转化成的Mat一个一个地存储到channels中  
        channels.push_back(channel);
        //在均值文件上移动一个通道
        data += mean_blob.height() * mean_blob.width();
    }

    /* Merge the separate channels into a single image. */
    cv::Mat mean;
    cv::merge(channels, mean);//将所有的通道合成为一张图

    /* Compute the global mean pixel value and create a mean image
     * filled with this value. */

    //求得均值文件的每个通道的平均值，记录在channel_mean中 
    cv::Scalar channel_mean = cv::mean(mean);
    //用上面求得的各个通道的平均值初始化mean_，作为数据集图像的均值  
    mean_ = cv::Mat(input_geometry_, mean.type(), channel_mean);
}

std::vector<float> Classifier::Predict(const cv::Mat& img) {
    //input_layer是网络的输入blob
    Blob<float>* input_layer = net_->input_blobs()[0];
    //表示网络只输入一张图像，图像的通道数是num_channels_，高为input_geometry_.height，宽为input_geometry_.width
    input_layer->Reshape(1, num_channels_,
                         input_geometry_.height, input_geometry_.width);
    /* Forward dimension change to all layers. */
    
    //初始化网络各层
    net_->Reshape();
    //存储输入图像的各个通道  
    std::vector<cv::Mat> input_channels;
    //将存储输入图像的各个通道的input_channels放入网络的输入blob中  
    WrapInputLayer(&input_channels);
    //将img的各通道分开并存储在input_channels中  
    Preprocess(img, &input_channels);

    //进行网络的前向传输
    net_->ForwardPrefilled();

    /* Copy the output layer to a std::vector */
    Blob<float>* output_layer = net_->output_blobs()[0];//output_layer指向网络输出的数据，存储网络输出数据的blob的规格是(1,c,1,1)  
    //begin指向输入数据对应的第一类的概率，end指向输入数据对应的最后一类的概率，返回输入数据经过网络前向计算后输出的对应于各个类的分数
    const float* begin = output_layer->cpu_data();
    const float* end = begin + output_layer->channels();
    return std::vector<float>(begin, end);
}

/* Wrap the input layer of the network in separate cv::Mat objects
 * (one per channel). This way we save one memcpy operation and we
 * don't need to rely on cudaMemcpy2D. The last preprocessing
 * operation will write the separate channels directly to the input
 * layer. */
void Classifier::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
    Blob<float>* input_layer = net_->input_blobs()[0];//input_layer指向网络输入的blob
    //得到网络指定输的宽、高
    int width = input_layer->width();
    int height = input_layer->height();
    float* input_data = input_layer->mutable_cpu_data();//input——data指向网络输入法让blob
    for (int i = 0; i < input_layer->channels(); ++i) {
        //将网络输入blob的数据同Mat关联起来  
        cv::Mat channel(height, width, CV_32FC1, input_data);
        //将上面的Mat同input_channels关联起来
        input_channels->push_back(channel);
        //一个一个通道地操作  
        input_data += width * height;
    }
}

void Classifier::Preprocess(const cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
    /* Convert the input image to the input image format of the network. */
    cv::Mat sample;
    if (img.channels() == 3 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGR2GRAY);
    else if (img.channels() == 4 && num_channels_ == 1)
        cv::cvtColor(img, sample, CV_BGRA2GRAY);
    else if (img.channels() == 4 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_BGRA2BGR);
    else if (img.channels() == 1 && num_channels_ == 3)
        cv::cvtColor(img, sample, CV_GRAY2BGR);
    else
        sample = img;//if-else嵌套表示了要将输入的img转化为num_channels_通道的  

    cv::Mat sample_resized;
    if (sample.size() != input_geometry_)
        cv::resize(sample, sample_resized, input_geometry_);//将输入图像的尺寸强制转化为网络规定的输入尺寸 
    else
        sample_resized = sample;

    cv::Mat sample_float;
    if (num_channels_ == 3)
        sample_resized.convertTo(sample_float, CV_32FC3);
    else
        sample_resized.convertTo(sample_float, CV_32FC1);//将输入图像转化为网络前传合法的数据规格

    cv::Mat sample_normalized;
    cv::subtract(sample_float, mean_, sample_normalized);//将图像减去均值

    /* This operation will write the separate BGR planes directly to the
     * input layer of the network because it is wrapped by the cv::Mat
     * objects in input_channels. */
    /*
    将减去均值的图像分散在input_channels中，由于在WrapInputLayer函数中， input_channels已经和网络的输入blob关联起来了，
    因此在这里实际上是把图像送入了网络的输入blob
    */ 
    cv::split(sample_normalized, *input_channels);
}

#endif
