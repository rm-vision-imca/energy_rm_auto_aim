
#include <string>
#include "energy_detector/detector.hpp"
#include "gtest/gtest.h"
using namespace rm_auto_aim;
// Nanodet-Plus OpenVino 测试
TEST(energy_detector, nanodet_test)
{
#ifndef VIDEO
#define VIDEO
#endif
#ifndef TEST_DIR
#define TEST_DIR
#endif
    Detector energy_search(BLUE);
    std::string video_path = std::string(TEST_DIR) + "/video/2xfile.mp4";
    cv::VideoCapture cap(video_path);

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        cv::Mat color_frame = frame.clone();
        std::vector<cv::Point2f> points;
        cv::Point2f center_to_image;
        auto leafs = energy_search.detect(frame);
        cv::imshow("frame", frame);
        energy_search.drawRuselt(color_frame);
        cv::imshow("color_frame", color_frame);
        cv::waitKey(10);

        if (!cap.read(frame))
            break;
    }
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
