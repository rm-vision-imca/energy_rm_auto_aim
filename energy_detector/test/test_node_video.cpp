
#include <string>
#include "energy_detector/detector.hpp"
#include "gtest/gtest.h"
#include <rclcpp/executors.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/utilities.hpp>
#include "energy_detector/energy_detector_node.hpp"
using namespace rm_auto_aim;
TEST(energy_detector, test_node_video)
{
#ifndef VIDEO
#define VIDEO
#endif
#ifndef TEST_DIR
#define TEST_DIR
#endif
    rclcpp::NodeOptions options;
    auto node = std::make_shared<rm_auto_aim::EnergyDetector>(options);
    std::string video_path = std::string(TEST_DIR) + "/video/2xfile.mp4";
    std::cout<<"读取视频中\n";
    cv::VideoCapture cap(video_path);
    std::cout<<"初始化成功\n";
    cv::Point2f cp;
    // auto predict_sub=node->create_subscription<geometry_msgs::msg::Point>("tracker/LeafTarget",rclcpp::SensorDataQoS(),[&](const geometry_msgs::msg::Point::ConstPtr msg){
    //     cp.x=msg->z;
    //     cp.y=msg->y;
    // });
    std::cout<<"初始化成功\n";
    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        cv::imshow("src",frame);
        cv::waitKey(10);
        cv::Mat color_frame = frame.clone();
        color_frame=node->VideoTest(color_frame);
        cv::imshow("color_frame",color_frame);
        cv::waitKey(10);
        cv::Mat predict_img=color_frame.clone();
        cv::circle(predict_img,cp,5,cv::Scalar(255,255,255),-1);
        cv::imshow("predict_img",predict_img);
        cv::waitKey(10);
        if (!cap.read(frame))
            break;
    }
    node.reset();
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    auto result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
