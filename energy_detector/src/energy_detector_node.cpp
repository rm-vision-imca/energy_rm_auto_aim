#include "energy_detector/energy_detector_node.hpp"
#include "geometry_msgs/msg/point.hpp"
namespace rm_auto_aim
{
  EnergyDetector::EnergyDetector(const rclcpp::NodeOptions &options)
      : Node("energy_detector", options)
  {

    RCLCPP_INFO(this->get_logger(), "<节点初始化> 能量机关检测器");
    // Detector
    detector_ = initDetector();
    RCLCPP_INFO(this->get_logger(), "<节点初始化> 检测器参数加载成功");
    // Visualization Marker Publisher 可视化
    // See http://wiki.ros.org/rviz/DisplayTypes/Marker
    leaf_marker_.ns = "leafs";
    leaf_marker_.action = visualization_msgs::msg::Marker::ADD;
    leaf_marker_.type = visualization_msgs::msg::Marker::CUBE; // 表示创建一个标记立方体
    leaf_marker_.scale.x = 0.05;                               // 缩放定义
    leaf_marker_.scale.z = 0.125;
    leaf_marker_.color.a = 1.0;
    leaf_marker_.color.g = 0.5;
    leaf_marker_.color.b = 1.0;
    leaf_marker_.lifetime = rclcpp::Duration::from_seconds(0.1); // 生命周期0.1s
    RCLCPP_INFO(this->get_logger(), "<节点初始化> 可视化参数加载成功");
    marker_pub_ =
        this->create_publisher<visualization_msgs::msg::MarkerArray>("/energy/detector/marker", 10);
    debug_ = this->declare_parameter("debug", false);
    if (debug_)
    {
      createDebugPublishers(); // 发布识别结果的image
    }

    // Debug param change moniter
    debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
    debug_cb_handle_ =
        debug_param_sub_->add_parameter_callback("debug", [this](const rclcpp::Parameter &p)
                                                 {
      debug_ = p.as_bool();
      debug_ ? createDebugPublishers() : destroyDebugPublishers(); });

    // 接受相机的消息
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info)
        {
          cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
          cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
          pnp_solver_ = std::make_unique<PnPSolver>(camera_info->k, camera_info->d);
          cam_info_sub_.reset();
        });

    // 相机视频接收者
    img_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/image_raw", rclcpp::SensorDataQoS(),
        std::bind(&EnergyDetector::ImageCallBack, this, std::placeholders::_1));
    leafs_pub_ = this->create_publisher<auto_aim_interfaces::msg::Leafs>("/detector/leafs", rclcpp::SensorDataQoS());
    RCLCPP_INFO(this->get_logger(), "<节点初始化> 能量机关检测器完成");
  }

  cv::Mat EnergyDetector::VideoTest(cv::Mat &img)
  {
    cv::Mat result_img = img.clone();
    auto leafs = detector_->detect(img);
    detector_->drawRuselt(result_img);
    auto_aim_interfaces::msg::Leaf leaf_msg;
    for (const auto &leaf : leafs)
    {
      // leaf info
      leaf_msg.leaf_center.x = 0;
      leaf_msg.leaf_center.y = leaf.kpt[5].y;
      leaf_msg.leaf_center.z = leaf.kpt[5].x;
      // R info
      leaf_msg.r_center.x = 0;
      leaf_msg.r_center.y = leaf.kpt[2].y;
      leaf_msg.r_center.z = leaf.kpt[2].x;
      leafs_msg_.leafs.emplace_back(leaf_msg);
    }
    leafs_pub_->publish(leafs_msg_);
    return result_img;
  }
  void EnergyDetector::ImageCallBack(const sensor_msgs::msg::Image::SharedPtr img_msg)
  {
    auto leafs = detectLeafs(img_msg);
    if (pnp_solver_ != nullptr)
    {
      leafs_msg_.header = img_msg->header;
      leafs_msg_.leafs.clear();
      leaf_marker_.id = 0;
      text_marker_.id = 0;

      auto_aim_interfaces::msg::Leaf leaf_msg;
      for (const auto &leaf : leafs)
      {
        // leaf info
        leaf_msg.leaf_center.x = 0;
        leaf_msg.leaf_center.y = leaf.kpt[5].y;
        leaf_msg.leaf_center.z = leaf.kpt[5].x;

        // R info
        leaf_msg.r_center.x = 0;
        leaf_msg.r_center.y = leaf.kpt[2].y;
        leaf_msg.r_center.z = leaf.kpt[2].x;

        cv::Mat rvec, tvec;
        bool success = pnp_solver_->solvePnP_(leaf, rvec, tvec);
        if (success)
        {
          // Fill pose
          leaf_msg.pose.position.x = tvec.at<double>(0);
          leaf_msg.pose.position.y = tvec.at<double>(1);
          leaf_msg.pose.position.z = tvec.at<double>(2);
          // rvec to 3x3 rotation matrix
          cv::Mat rotation_matrix;
          cv::Rodrigues(rvec, rotation_matrix);
          // rotation matrix to quaternion
          tf2::Matrix3x3 tf2_rotation_matrix(
              rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1),
              rotation_matrix.at<double>(0, 2), rotation_matrix.at<double>(1, 0),
              rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2),
              rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1),
              rotation_matrix.at<double>(2, 2));
          tf2::Quaternion tf2_q;
          tf2_rotation_matrix.getRotation(tf2_q);
          leaf_msg.pose.orientation = tf2::toMsg(tf2_q);

          // Fill the distance to image center
          leaf_msg.distance_to_image_center = pnp_solver_->calculateDistanceToCenter(leaf.kpt[5]);
          // Fill the markers
          leaf_marker_.id++;
          leaf_marker_.scale.y = 0.332;
          leaf_marker_.pose = leaf_msg.pose;
          text_marker_.id++;
          text_marker_.pose.position = leaf_msg.pose.position;
          text_marker_.pose.position.y -= 0.1;
          leafs_msg_.leafs.emplace_back(leaf_msg);
          marker_array_.markers.emplace_back(leaf_marker_);
          marker_array_.markers.emplace_back(text_marker_);
        }
        else
        {
          RCLCPP_WARN(this->get_logger(), "PnP failed!");
        }
        // Publishing detected leafs
        leafs_pub_->publish(leafs_msg_);

        // Publishing marker
        publishMarkers();
      }
    }
  }

  void EnergyDetector::createDebugPublishers()
  {
    leafs_data_pub = this->create_publisher<auto_aim_interfaces::msg::DebugLeafs>("/detector/debug_leafs", 10);

    result_img_pub_ = image_transport::create_publisher(this, "/detector/leaf_result_img");
  }

  void EnergyDetector::destroyDebugPublishers()
  {
    leafs_data_pub.reset();
    result_img_pub_.shutdown();
  }

  std::unique_ptr<Detector> EnergyDetector::initDetector()
  {
    rcl_interfaces::msg::ParameterDescriptor param_desc;
    param_desc.description = "0-RED, 1-BLUE";
    param_desc.integer_range.resize(1);
    param_desc.integer_range[0].step = 1;
    param_desc.integer_range[0].from_value = 0;
    param_desc.integer_range[0].to_value = 1;
    RCLCPP_INFO(this->get_logger(), "<节点初始化> 检测器参数加载中");
    auto detect_color = declare_parameter("detect_color", BLUE, param_desc);
    auto detector = std::make_unique<Detector>(detect_color);
    return detector;
  }

  std::vector<Leaf> EnergyDetector::detectLeafs(
      const sensor_msgs::msg::Image::ConstSharedPtr &img_msg)
  { // Convert ROS img to cv::Mat
    auto img = cv_bridge::toCvShare(img_msg, "rgb8")->image;
    // Update params
    detector_->detect_color = get_parameter("detect_color").as_int();
    auto leafs = detector_->detect(img); // 识别

    auto final_time = this->now();
    auto latency = (final_time - img_msg->header.stamp).seconds() * 1000;
    RCLCPP_DEBUG_STREAM(this->get_logger(), "Latency: " << latency << "ms"); // 计算图像处理的延迟输出到日志中
    // Publish debug info
    if (debug_)
    {
      leafs_data_pub->publish(detector_->debug_leafs);
      detector_->drawRuselt(img);
      // Draw camera center
      cv::circle(img, cam_center_, 5, cv::Scalar(255, 0, 0), 2);
      // Draw latency
      std::stringstream latency_ss;
      latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency << "ms";
      auto latency_s = latency_ss.str();
      cv::putText(
          img, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
      result_img_pub_.publish(cv_bridge::CvImage(img_msg->header, "rgb8", img).toImageMsg());
    }
    return leafs;
  }

  void EnergyDetector::publishMarkers()
  {
    using Marker = visualization_msgs::msg::Marker;
    leaf_marker_.action = leafs_msg_.leafs.empty() ? Marker::DELETE : Marker::ADD;
    marker_array_.markers.emplace_back(leaf_marker_);
    marker_pub_->publish(marker_array_);
  }

} // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::EnergyDetector)