#ifndef DETECTOR_HPP_
#define DETECTOR_HPP_

#include <sys/types.h>

#include <cassert>
#include <cfloat>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <inference_engine.hpp>
#include <iostream>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <sstream>
#include <string>
#include <vector>

#include "energy_detector/Leaf.hpp"
#include "auto_aim_interfaces/msg/debug_leafs.hpp"
using namespace InferenceEngine;

#define pi 3.1415926
#define NMS_THRESHOLD 0.10f  // NMS参数
#define CONF_THRESHOLD 0.40f // 置信度参数
#define CONF_REMAIN 0.0      // 保留一帧保留的权重比例，如果不保留填写为0
#define IMG_SIZE 416         // 推理图像大小，如果不是640 和 416 需要自己在下面添加anchor
#define ANCHOR 3             // anchor 数量
#define DETECT_MODE 1        // ARMOR 0 WIN 1 BOARD 2
#define DEVICE "CPU"         // 设备选择
#define VIDEO                // 是否展示推理视频

#if DETECT_MODE == 0 // 装甲板四点模型
#define KPT_NUM 4
#define CLS_NUM 14
#define MODEL_PATH ""
#elif DETECT_MODE == 1 // 能量机关五点模型
#define KPT_NUM 5
#define CLS_NUM 4
#ifdef CURRENT_PKG_DIR
#define MODEL_PATH CURRENT_PKG_DIR "/models/energy.xml"
#endif
#elif DETECT_MODE == 2 // 视觉识别版检测模型
#define KPT_NUM 0
#define CLS_NUM 4
#define MODEL_PATH ""
#endif

namespace rm_auto_aim
{
  class Detector
  {
  public:
    Detector(const int& detect_color);
    std::vector<Leaf> detect(const cv::Mat &input);

    cv::Mat letter_box(cv::Mat &src, int h, int w, std::vector<float> &padd);

    std::vector<cv::Point2f> scale_box_kpt(
        std::vector<cv::Point2f> points, std::vector<float> &padd, float raw_w, float raw_h, int idx);

    cv::Rect scale_box(cv::Rect box, std::vector<float> &padd, float raw_w, float raw_h);

    void drawPred(
        int classId, float conf, cv::Rect box, std::vector<cv::Point2f> point, cv::Mat &frame,
        const std::vector<std::string> &classes);

    static void generate_proposals(int stride, const float *feat, std::vector<Leaf> &Leafs);

    std::vector<Leaf> work(cv::Mat src_img, uint8_t detect_color);
    std::vector<Leaf> Leaf_filter(
        std::vector<Leaf> &leafs, const int MAX_WIDTH, const int MAX_HEIGHT);
    void drawRuselt(cv::Mat &src);
    // Debug msgs
    cv::Mat result_img;
    auto_aim_interfaces::msg::DebugLeafs debug_leafs;
    std::vector<Leaf> leafs_;
    int detect_color;
  private:
    cv::Mat dilate_struct;
    cv::Mat erode_struct;
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;
    ov::Tensor input_tensor1;
#if DETECT_MODE == 0
    const std::vector<std::string> class_names = {"B1", "B2", "B3", "B4", "B5", "BO", "BS",
                                                  "R1", "R2", "R3", "R4", "R5", "RO", "RS"};
#elif DETECT_MODE == 1
    const std::vector<std::string> class_names = {"RR", "RW", "BR", "BW"};
#elif DETECT_MODE == 2
    const std::vector<std::string> class_names = {"RA", "RD", "BA", "BD"};
#endif

    static float sigmoid(float x) { return static_cast<float>(1.f / (1.f + exp(-x))); }
  };
} // namespace rm_auto_aim
#endif // GMASTER_CV_2023_ARMORNEWYOLO_H