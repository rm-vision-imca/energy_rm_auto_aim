import rclpy
import cv2
import matplotlib.pyplot as plt
import numpy as np
from auto_aim_interfaces.srv import TrackingMode
from auto_aim_interfaces.msg import Leafs
from .utils.angleProcessor import bigPredictor, smallPredictor, angleObserver, trans, clock, mode
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
deltaT = 0.2

class energy_tracker(Node):
    def __init__(self,options):
        super().__init__(options)
        self.get_logger().info("<节点初始化> 能量机关预测器")
        self.predict_mode_service=self.create_service(TrackingMode, "Predict_mode", self.predict_mode_service_callback)#预测模式服务端
        self.get_logger().info("<节点初始化> 能量机关预测器/预测模式服务端")
        self.Leafs_Sub=self.create_subscription(Leafs, "detector/leafs", self.LeafsCallback,rclpy.qos.qos_profile_sensor_data)
        self.Target_pub=self.create_publisher(Point, "tracker/LeafTarget",rclpy.qos.qos_profile_sensor_data)
        self.is_start=True
        self.moveMode = mode.big
        self.freq = 50 #看看能否从ros传过去
        self.angles = []
        self.xy = []
        self.observer = angleObserver(clockMode=clock.clockwise)
        self.Target=Point()
    def predict_mode_service_callback(self,mode_request,mode_response):
        if mode_request.mode==1:
            self.moveMode=mode.small
        elif mode_request.mode==2:
            self.moveMode=mode.big
        elif mode_request.mode==0:
            self.moveMode=mode.person
        self.is_start=True
        mode_response.success=True
        return mode_response
     
    def LeafsCallback(self,leafs_msg):
        for leaf in leafs_msg.leafs:
            if self.is_start is True:
                if self.moveMode == mode.small:
                    self.predictor = smallPredictor(freq=self.freq, deltaT=deltaT)
                elif self.moveMode == mode.big:
                    self.predictor = bigPredictor(freq=self.freq, deltaT=deltaT)
                interval = int(self.freq * deltaT)
                self.is_start=False
                
            A_p=np.array([leaf.leaf_center.z,leaf.leaf_center.y])
            R_p=np.array([leaf.r_center.z,leaf.r_center.y]) 
            x, y = A_p - R_p # 分别算出二维r中心与扇叶中心的x,y距离
            self.radius=np.sqrt(x**2+y**2)
            angle = self.observer.update(x, y, self.radius)#角度更新
            self.get_logger().info("angle={},x={},y={},r={}".format(angle, x, y, self.radius))
            self.angles.append(angle)#角度添加
            flag, deltaAngle = self.predictor.update(angle)
            if flag:
                angle = trans(x, y) + deltaAngle
                x = np.cos(angle) * self.radius  # 提前x 秒后的扇叶中心的x
                y = np.sin(angle) * self.radius  # 提前x 秒后的扇叶中心的y
                x, y = np.array([x, y]) + R_p  # 得到最终的预测扇叶中心
                if not np.isnan(angle): 
                    self.xy.append([x, y])  # 加入xy列表加入x,y的元组
                    self.Target.y=y
                    self.Target.z=x
                    self.Target_pub.publish(self.Target)
            
def main(args=None):
    rclpy.init(args=args)
    node=energy_tracker("energy_tracker_node")
    rclpy.spin(node)
    rclpy.shutdown()
        

    
        