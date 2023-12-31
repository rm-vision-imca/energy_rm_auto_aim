#include <string>
#include "energy_detector/detector.hpp"
#include "gtest/gtest.h"
#include <rclcpp/executors.hpp>
#include <rclcpp/node_options.hpp>
#include <rclcpp/utilities.hpp>
#include "energy_detector/energy_detector_node.hpp"
#include "auto_aim_interfaces/msg/tracker2_d.hpp"
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
    cv::Point2f pre_Point;
    auto node = std::make_shared<rm_auto_aim::EnergyDetector>(options);
    auto Test_node = std::make_shared<rclcpp::Node>("Test_node");
    auto Test_Sub = Test_node->create_subscription<auto_aim_interfaces::msg::Tracker2D>("tracker/LeafTarget", 10, [&](const auto_aim_interfaces::msg::Tracker2D::SharedPtr Point)
                                                                                        {
        RCLCPP_INFO(Test_node->get_logger(),"x:%.2f y:%.2f",Point->x,Point->y);
        pre_Point.x=Point->x;
        pre_Point.y=Point->y;
        std::cout<<Point->x<<" "<<Point->y<<std::endl; });
    std::string video_path = std::string(TEST_DIR) + "/video/2xfile.mp4";
    std::cout << "读取视频中\n";
    cv::VideoCapture cap(video_path);
    std::cout << "初始化成功\n";

    while (true)
    {
        rclcpp::spin_some(Test_node);
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
            break;
        cv::imshow("src", frame);
        cv::waitKey(10);
        cv::Mat color_frame = frame.clone();
        color_frame = node->VideoTest(color_frame);
        cv::circle(color_frame, pre_Point, 5, cv::Scalar(255, 255, 255), -1);
        cv::imshow("color_frame", color_frame);
        cv::waitKey(10);
        if (!cap.read(frame))
            break;
    }
}
int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    rclcpp::init(argc, argv);
    auto result = RUN_ALL_TESTS();
    rclcpp::shutdown();
    return result;
}
