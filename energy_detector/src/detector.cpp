#include "detector.hpp"

namespace rm_auto_aim
{
  /*
      @param T length
      @param T1 cv_point_type
  */

  std::vector<Leaf> Detector::detect(const cv::Mat &input)
  {
    result_img = input.clone();
    auto leafs = work(result_img, this->detect_color);
    leafs_ = Leaf_filter(leafs, input.cols, input.rows);
    return leafs_;
  }

  Detector::Detector(const int &detect_color) : detect_color(detect_color)
  {
    model = core.read_model(MODEL_PATH);
    std::shared_ptr<ov::Model> model = core.read_model(MODEL_PATH);
    compiled_model = core.compile_model(model, DEVICE);
    std::map<std::string, std::string> config = {
        {InferenceEngine::PluginConfigParams::KEY_PERF_COUNT,
         InferenceEngine::PluginConfigParams::YES}};
    infer_request = compiled_model.create_infer_request();
    input_tensor1 = infer_request.get_input_tensor(0);
  }

  cv::Mat Detector::letter_box(cv::Mat &src, int h, int w, std::vector<float> &padd)
  {
    int in_w = src.cols;
    int in_h = src.rows;
    int tar_w = w;
    int tar_h = h;
    float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    int inside_w = round(in_w * r);
    int inside_h = round(in_h * r);
    int padd_w = tar_w - inside_w;
    int padd_h = tar_h - inside_h;
    cv::Mat resize_img;
    resize(src, resize_img, cv::Size(inside_w, inside_h));
    padd_w = padd_w / 2;
    padd_h = padd_h / 2;
    padd.push_back(padd_w);
    padd.push_back(padd_h);
    padd.push_back(r);
    int top = int(round(padd_h - 0.1));
    int bottom = int(round(padd_h + 0.1));
    int left = int(round(padd_w - 0.1));
    int right = int(round(padd_w + 0.1));
    copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
    return resize_img;
  }

  cv::Rect Detector::scale_box(cv::Rect box, std::vector<float> &padd, float raw_w, float raw_h)
  {
    cv::Rect scaled_box;
    scaled_box.width = box.width / padd[2];
    scaled_box.height = box.height / padd[2];
    scaled_box.x = std::max(std::min((float)((box.x - padd[0]) / padd[2]), (float)(raw_w - 1)), 0.f);
    scaled_box.y = std::max(std::min((float)((box.y - padd[1]) / padd[2]), (float)(raw_h - 1)), 0.f);
    return scaled_box;
  }

  std::vector<cv::Point2f> Detector::scale_box_kpt(
      std::vector<cv::Point2f> points, std::vector<float> &padd, float raw_w, float raw_h, int idx)
  {
    std::vector<cv::Point2f> scaled_points;
    for (int ii = 0; ii < KPT_NUM; ii++)
    {
      points[idx * KPT_NUM + ii].x = std::max(
          std::min((points[idx * KPT_NUM + ii].x - padd[0]) / padd[2], (float)(raw_w - 1)), 0.f);
      points[idx * KPT_NUM + ii].y = std::max(
          std::min((points[idx * KPT_NUM + ii].y - padd[1]) / padd[2], (float)(raw_h - 1)), 0.f);
      scaled_points.push_back(points[idx * KPT_NUM + ii]);
    }
    return scaled_points;
  }

  void Detector::drawPred(
      int classId, float conf, cv::Rect box, std::vector<cv::Point2f> point, cv::Mat &frame,
      const std::vector<std::string> &classes)
  { // 画图部分&得到点
    float x0 = box.x;
    float y0 = box.y;
    float x1 = box.x + box.width;
    float y1 = box.y + box.height;
#ifdef VIDEO
    cv::rectangle(frame, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(255, 255, 255), 1);
#endif
    cv::Point2f keypoints_center(0, 0);
    std::vector<bool> valid_keypoints(5, false);
    for (int i = 0; i < point.size(); i++)
    {
      if (i != 2 && point[i].x != 0 && point[i].y != 0)
      {
        valid_keypoints[i] = true;
      }
    }
    // 四种情况判断
    if (valid_keypoints[0] && valid_keypoints[1] && valid_keypoints[3] && valid_keypoints[4])
    {
      // 1. 四个关键点都有效，直接取中心点
      keypoints_center = (point[0] + point[1] + point[3] + point[4]) * 0.25;
    }
    else if (
        valid_keypoints[0] && valid_keypoints[3] && (!valid_keypoints[1] || !valid_keypoints[4]))
    {
      // 2. 0 3关键点有效，1 4 关键点缺少一个以上： 算 0 3 关键点的中点
      keypoints_center = (point[0] + point[3]) * 0.5;
    }
    else if (
        valid_keypoints[1] && valid_keypoints[4] && (!valid_keypoints[0] || !valid_keypoints[3]))
    {
      // 3. 1 4关键点有效，0 3 关键点缺少一个以上： 算 1 4 关键点的中点
      keypoints_center = (point[1] + point[4]) * 0.5;
    }
    else
    {
      // 4. 以上三个都不满足，算bbox中心点
      keypoints_center = cv::Point2f(x0 + box.width / 2, y0 + box.height / 2);
    }
#ifdef VIDEO
    cv::circle(frame, keypoints_center, 2, cv::Scalar(255, 255, 255), 2);
    for (int i = 0; i < KPT_NUM; i++)
      if (DETECT_MODE == 1)
        if (i == 2)
          cv::circle(frame, point[i], 4, cv::Scalar(163, 164, 163), 4);
        else
          cv::circle(frame, point[i], 3, cv::Scalar(0, 255, 0), 3);

    std::string label = cv::format("%.2f", conf);
    if (!classes.empty())
    {
      CV_Assert(classId < (int)classes.size());
      label = classes[classId] + ": " + label;
    }
    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseLine);
    y0 = std::max(int(y0), labelSize.height);
    cv::rectangle(
        frame, cv::Point(x0, y0 - round(1.5 * labelSize.height)),
        cv::Point(x0 + round(2 * labelSize.width), y0 + baseLine), cv::Scalar(255, 255, 255),
        cv::FILLED);
    cv::putText(frame, label, cv::Point(x0, y0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1.5);
#endif
    this->result_img = frame.clone();
  }

  void Detector::generate_proposals(int stride, const float *feat, std::vector<Leaf> &Leafs)
  { // 后处理部分，重做检测层
    int feat_w = IMG_SIZE / stride;
    int feat_h = IMG_SIZE / stride;
#if IMG_SIZE == 640
    float anchors[18] = {11, 10, 19, 15, 28, 22, 39, 34, 64,
                         48, 92, 76, 132, 110, 197, 119, 265, 162}; // 6 4 0
#elif IMG_SIZE == 416
    float anchors[18] = {26, 27, 28, 28, 27, 30, 29, 29, 29,
                         32, 30, 31, 30, 33, 32, 32, 32, 34}; // 4 1 6
#elif IMG_SIZE == 320
    float anchors[18] = {6, 5, 9, 7, 13, 9, 18, 15, 30,
                         23, 46, 37, 60, 52, 94, 56, 125, 72}; // 3 2 0
#endif
    int anchor_group = 0;
    if (stride == 8)
      anchor_group = 0;
    if (stride == 16)
      anchor_group = 1;
    if (stride == 32)
      anchor_group = 2;

    for (int anchor = 0; anchor < ANCHOR; anchor++)
    { // 对每个anchor进行便利
      for (int i = 0; i < feat_h; i++)
      { // self.grid[i][..., 0:1]
        for (int j = 0; j < feat_w; j++)
        { // self.grid[i][..., 1:2]
          // 每个tensor包含的数据是[x,y,w,h,conf,cls1pro,cls2pro,...clsnpro,kpt1.x,kpt1.y,kpt1.conf,kpt2...kptm.conf]
          // 一共的长度应该是 (5 + CLS_NUM + KPT_NUM * 3)
          float box_prob = feat
              [anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
               i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) + j * (5 + CLS_NUM + KPT_NUM * 3) + 4];
          box_prob = sigmoid(box_prob);
          if (box_prob < CONF_THRESHOLD)
            continue; // 删除置信度低的bbox

          float kptx[5], kpty[5], kptp[5];
          // xi,yi,pi 是每个关键点的xy坐标和置信度,最新的代码用不到pi,但是用户可以根据自己需求添加
          float x = feat
              [anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
               i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) + j * (5 + CLS_NUM + KPT_NUM * 3) + 0];
          float y = feat
              [anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
               i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) + j * (5 + CLS_NUM + KPT_NUM * 3) + 1];
          float w = feat
              [anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
               i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) + j * (5 + CLS_NUM + KPT_NUM * 3) + 2];
          float h = feat
              [anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
               i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) + j * (5 + CLS_NUM + KPT_NUM * 3) + 3];
          if (KPT_NUM != 0)
            for (int k = 0; k < KPT_NUM; k++)
            {
              kptx[k] = feat
                  [anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                   i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) + j * (5 + CLS_NUM + KPT_NUM * 3) + 5 +
                   CLS_NUM + k * 3];
              kpty[k] = feat
                  [anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                   i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) + j * (5 + CLS_NUM + KPT_NUM * 3) + 5 +
                   CLS_NUM + k * 3 + 1];
              kptp[k] = feat
                  [anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                   i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) + j * (5 + CLS_NUM + KPT_NUM * 3) + 5 +
                   CLS_NUM + k * 3 + 2];

              // 对关键点进行后处理(python 代码)
              // x_kpt[..., 0::3] = (x_kpt[..., ::3] * 2. - 0.5 + kpt_grid_x.repeat(1, 1, 1, 1, self.nkpt)) * self.stride[i]  # xy
              // x_kpt[..., 1::3] = (x_kpt[..., 1::3] * 2. - 0.5 + kpt_grid_y.repeat(1, 1, 1, 1, self.nkpt)) * self.stride[i]  # xy
              kptx[k] = (kptx[k] * 2 - 0.5 + j) * stride;
              kpty[k] = (kpty[k] * 2 - 0.5 + i) * stride;
            }
          double max_prob = 0;
          int idx = 0;
          for (int k = 5; k < CLS_NUM + 5; k++)
          {
            double tp = feat
                [anchor * feat_h * feat_w * (5 + CLS_NUM + KPT_NUM * 3) +
                 i * feat_w * (5 + CLS_NUM + KPT_NUM * 3) + j * (5 + CLS_NUM + KPT_NUM * 3) + k];
            tp = sigmoid(tp);
            if (tp > max_prob)
              max_prob = tp, idx = k;
          }
          float cof = std::min(box_prob * max_prob, 1.0);
          if (cof < CONF_THRESHOLD)
            continue;

          // xywh的后处理(python 代码)
          // xy = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
          // wh = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i].view(1, self.na, 1, 1, 2)  # wh
          x = (sigmoid(x) * 2 - 0.5 + j) * stride;
          y = (sigmoid(y) * 2 - 0.5 + i) * stride;
          w = pow(sigmoid(w) * 2, 2) * anchors[anchor_group * 6 + anchor * 2];
          h = pow(sigmoid(h) * 2, 2) * anchors[anchor_group * 6 + anchor * 2 + 1];
          // 将中心点变为左上点，转换为OpenCV rect类型
          float r_x = x - w / 2;
          float r_y = y - h / 2;
          Leaf obj;
          obj.rect.x = r_x;
          obj.rect.y = r_y;
          obj.rect.width = w;
          obj.rect.height = h;
          obj.label = idx - 5;
          obj.prob = cof;
          if (KPT_NUM != 0)
            for (int k = 0; k < KPT_NUM; k++)
              if (k != 2 && kptx[k] > r_x && kptx[k] < r_x + w && kpty[k] > r_y && kpty[k] < r_y + h)
                obj.kpt.push_back(cv::Point2f(kptx[k], kpty[k]));
              else if (k == 2)
                obj.kpt.push_back(cv::Point2f(kptx[k], kpty[k]));
              else
                obj.kpt.push_back(cv::Point2f(0, 0));
          Leafs.push_back(obj);
        }
      }
    }
  }

  std::vector<Leaf> Detector::work(cv::Mat src_img, uint8_t detect_color)
  {
    this->debug_leafs.data.clear();
    int attack_color;
    if (detect_color == BLUE)
      attack_color = 2;
    else
      attack_color = 0;
    int img_h = IMG_SIZE;
    int img_w = IMG_SIZE;
    cv::Mat img;
    std::vector<float> padd;
    cv::Mat boxed = letter_box(src_img, img_h, img_w, padd);
    cv::cvtColor(boxed, img, cv::COLOR_BGR2RGB);
    auto data1 = input_tensor1.data<float>();
    for (int h = 0; h < img_h; h++)
    {
      for (int w = 0; w < img_w; w++)
      {
        for (int c = 0; c < 3; c++)
        {
          int out_index = c * img_h * img_w + h * img_w + w;
          data1[out_index] = float(img.at<cv::Vec3b>(h, w)[c]) / 255.0f;
        }
      }
    }
    //    infer_request.infer(); //推理并获得三个提取头
    infer_request.start_async();
    infer_request.wait();
    auto output_tensor_p8 = infer_request.get_output_tensor(0);
    const float *result_p8 = output_tensor_p8.data<const float>();
    auto output_tensor_p16 = infer_request.get_output_tensor(1);
    const float *result_p16 = output_tensor_p16.data<const float>();
    auto output_tensor_p32 = infer_request.get_output_tensor(2);
    const float *result_p32 = output_tensor_p32.data<const float>();
    std::vector<Leaf> proposals;
    std::vector<Leaf> Leafs8;
    std::vector<Leaf> Leafs16;
    std::vector<Leaf> Leafs32;
    generate_proposals(8, result_p8, Leafs8);
    proposals.insert(proposals.end(), Leafs8.begin(), Leafs8.end());
    generate_proposals(16, result_p16, Leafs16);
    proposals.insert(proposals.end(), Leafs16.begin(), Leafs16.end());
    generate_proposals(32, result_p32, Leafs32);
    proposals.insert(proposals.end(), Leafs32.begin(), Leafs32.end());
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    std::vector<cv::Point2f> points;
    for (size_t i = 0; i < proposals.size(); i++)
    {
      classIds.push_back(proposals[i].label);
      confidences.push_back(proposals[i].prob);
      boxes.push_back(proposals[i].rect);
      for (auto ii : proposals[i].kpt)
        points.push_back(ii);
    }
    std::vector<int> picked;
    std::vector<float> picked_useless; // SoftNMS
    std::vector<Leaf> Leaf_result;

    // SoftNMS 要求OpenCV>=4.6.0
    //     cv::dnn::softNMSBoxes(boxes, confidences, picked_useless, CONF_THRESHOLD, NMS_THRESHOLD, picked);
    cv::dnn::NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD, picked);
    for (size_t i = 0; i < picked.size(); i++)
    {
      cv::Rect scaled_box = scale_box(boxes[picked[i]], padd, src_img.cols, src_img.rows);
      std::vector<cv::Point2f> scaled_point;
      if (KPT_NUM != 0)
        scaled_point = scale_box_kpt(points, padd, src_img.cols, src_img.rows, picked[i]);
      Leaf obj;
      obj.rect = scaled_box;
      obj.label = classIds[picked[i]];
      obj.prob = confidences[picked[i]];
      if (KPT_NUM != 0)
        obj.kpt = scaled_point;
      if (DETECT_MODE == 1 && classIds[picked[i]] == attack_color)
      {
        auto_aim_interfaces::msg::DebugLeaf leaf_data;
        // TOP_LEFT
        leaf_data.left_top.x = 0;
        leaf_data.left_top.z = obj.kpt.at(LeafPointType::TOP_LEFT).x;
        leaf_data.left_top.y = obj.kpt.at(LeafPointType::TOP_LEFT).y;
        // BOTTOM_LEFT
        leaf_data.left_bottom.x = 0;
        leaf_data.left_bottom.z = obj.kpt.at(LeafPointType::BOTTOM_LEFT).x;
        leaf_data.left_bottom.y = obj.kpt.at(LeafPointType::BOTTOM_LEFT).y;
        // BOTTOM_RIGHT
        leaf_data.right_bottom.x = 0;
        leaf_data.right_bottom.z = obj.kpt.at(LeafPointType::BOTTOM_RIGHT).x;
        leaf_data.right_bottom.y = obj.kpt.at(LeafPointType::BOTTOM_RIGHT).y;
        // TOP_RIGHT
        leaf_data.right_top.x = 0;
        leaf_data.right_top.z = obj.kpt.at(LeafPointType::TOP_RIGHT).x;
        leaf_data.right_top.y = obj.kpt.at(LeafPointType::TOP_RIGHT).y;
        // R
        leaf_data.r_center.x = 0;
        leaf_data.r_center.z = obj.kpt.at(LeafPointType::R).x;
        leaf_data.r_center.y = obj.kpt.at(LeafPointType::R).y;
        debug_leafs.data.emplace_back(leaf_data);
        Leaf_result.push_back(obj);

        // drawPred(
        //     classIds[picked[i]], confidences[picked[i]], scaled_box, scaled_point, src_img,
        //     class_names);
      }
    }
    return Leaf_result;
  }
  std::vector<Leaf> Detector::Leaf_filter(
      std::vector<Leaf> &leafs, const int MAX_WIDTH, const int MAX_HEIGHT)
  {
    float angle;
    auto Get_Point = [&](cv::Point2f pt1, cv::Point2f pt2) -> std::vector<cv::Point2f>
    {
      std::vector<cv::Point2f> Point_2;
      cv::Point2f center = (pt1 + pt2) * 0.5f;
      float width = std::abs(pt2.x - pt1.x);
      float height = std::abs(pt2.y - pt1.y);
      angle = std::atan2(pt2.y - pt1.y, pt2.x - pt1.x);
      angle = angle * 180.0f / CV_PI;
      //std::cout<<"angle"<<angle<<std::endl;

      cv::Mat rotationMatrix = cv::getRotationMatrix2D(center, angle, 1);
      std::vector<cv::Point2f> points;
      points.push_back(pt1);
      points.push_back(pt2);

      // 逆向旋转已知的两个对角点
      std::vector<cv::Point2f> rotatedPoints;
      cv::transform(points, rotatedPoints, rotationMatrix);
      Point_2.emplace_back(rotatedPoints[0]);
      Point_2.emplace_back(rotatedPoints[1]);
      return Point_2;
    };
    std::vector<Leaf> result;
    for (auto &leaf : leafs)
    {
      float x0 = leaf.rect.x;
      float y0 = leaf.rect.y;
      float x1 = leaf.rect.x + leaf.rect.width;
      float y1 = leaf.rect.y + leaf.rect.height;
      cv::Point2f keypoints_center(0, 0);
      std::vector<bool> valid_keypoints(5, false);
      for (int i = 0; i < leaf.kpt.size(); i++)
      {
        if (i != 2 && leaf.kpt[i].x != 0 && leaf.kpt[i].y != 0)
        {
          valid_keypoints[i] = true;
        }
      }
      std::vector<cv::Point2f> pair_point;
      // 四种情况判断
      if (valid_keypoints[0] && valid_keypoints[1] && valid_keypoints[3] && valid_keypoints[4])
      {
        // 1. 四个关键点都有效，直接取中心点
        keypoints_center = (leaf.kpt[0] + leaf.kpt[1] + leaf.kpt[3] + leaf.kpt[4]) * 0.25;
      }
      else if (
          valid_keypoints[0] && valid_keypoints[3] && (!valid_keypoints[1] || !valid_keypoints[4]))
      {
        // 2. 0 3关键点有效，1 4 关键点缺少一个以上： 算 0 3 关键点的中点
        keypoints_center = (leaf.kpt[0] + leaf.kpt[3]) * 0.5;
        pair_point = Get_Point(leaf.kpt[0], leaf.kpt[3]);
        if (angle > 0 && angle < 180)
        {
          leaf.kpt[1] = valid_keypoints[1]?leaf.kpt[1]:pair_point.at(0);
          leaf.kpt[4] = valid_keypoints[4]?leaf.kpt[4]:pair_point.at(1);
        }
        else
        {
          leaf.kpt[1] = valid_keypoints[1]?leaf.kpt[1]:pair_point.at(1);
          leaf.kpt[4] = valid_keypoints[4]?leaf.kpt[4]:pair_point.at(0);
        }
      }
      else if (
          valid_keypoints[1] && valid_keypoints[4] && (!valid_keypoints[0] || !valid_keypoints[3]))
      {
        // 3. 1 4关键点有效，0 3 关键点缺少一个以上： 算 1 4 关键点的中点
        keypoints_center = (leaf.kpt[1] + leaf.kpt[4]) * 0.5;
        pair_point = Get_Point(leaf.kpt[1], leaf.kpt[4]);
        if (angle > 0 && angle < 180)
        {
          leaf.kpt[0] = valid_keypoints[0]?leaf.kpt[0]:pair_point.at(0);
          leaf.kpt[3] = valid_keypoints[3]?leaf.kpt[3]:pair_point.at(1);
        }
        else
        {
          leaf.kpt[0] = valid_keypoints[0]?leaf.kpt[0]:pair_point.at(1);
          leaf.kpt[3] = valid_keypoints[3]?leaf.kpt[3]:pair_point.at(0);
        }
      }
      else
      {
        // 4. 以上三个都不满足，算bbox中心点
        keypoints_center = cv::Point2f(x0 + leaf.rect.width / 2, y0 + leaf.rect.height / 2);
      }
      leaf.kpt.emplace_back(keypoints_center);
      LeafType type;
      for (size_t i = 0; i < leaf.kpt.size(); i++)
      {
        if (
            leaf.kpt[i].x < 0 or leaf.kpt[i].x > MAX_WIDTH or leaf.kpt[i].y < 0 or
            leaf.kpt[i].x > MAX_HEIGHT or leaf.kpt[i].x < 0 or leaf.kpt[i].x > MAX_WIDTH or
            leaf.kpt[i].y < 0 or leaf.kpt[i].y > MAX_HEIGHT)
        {

          type = LeafType::INVALID;
          break;
        }
        type = LeafType::VALID;
      }
      if (type == LeafType::VALID)
        result.emplace_back(leaf);
    }
    return result;
  }
  void Detector::drawRuselt(cv::Mat &src)
  {
    for (auto &leaf : leafs_)
    {
      float x0 = leaf.rect.x;
      float y0 = leaf.rect.y;
      float x1 = leaf.rect.x + leaf.rect.width;
      float y1 = leaf.rect.y + leaf.rect.height;
      int baseLine;
      float prob = leaf.prob;
      std::string label = cv::format("%.2f", prob);
      cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.25, 1, &baseLine);
      y0 = std::max(int(y0), labelSize.height);
      cv::rectangle(
          src, cv::Point(x0, y0 - round(1.5 * labelSize.height)),
          cv::Point(x0 + round(2 * labelSize.width), y0 + baseLine), cv::Scalar(255, 255, 255),
          cv::FILLED);
      cv::putText(src, label, cv::Point(x0, y0), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(), 1.5);
      cv::rectangle(src, leaf.rect, cv::Scalar(255, 255, 0), 3);
      int i=0;
      for (auto &p : leaf.kpt)
      {
        cv::circle(src, p, 5, cv::Scalar(255, 0, 0), 3);
        // if(i++==4)cv::circle(src, p, 5, cv::Scalar(255, 255, 255), 3);
      }
    }
  }
} // namespace rm_auto_aim