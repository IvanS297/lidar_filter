import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

import numpy as np
import math
from .glass_filter import GlassFilter

class filter(Node):
    def __init__(self, node_name, *, context = None, cli_args = None, namespace = None, use_global_arguments = True, enable_rosout = True, start_parameter_services = True, parameter_overrides = None, allow_undeclared_parameters = False, automatically_declare_parameters_from_overrides = False, enable_logger_service = False):
        super().__init__(node_name, context=context, cli_args=cli_args, namespace=namespace, use_global_arguments=use_global_arguments, enable_rosout=enable_rosout, start_parameter_services=start_parameter_services, parameter_overrides=parameter_overrides, allow_undeclared_parameters=allow_undeclared_parameters, automatically_declare_parameters_from_overrides=automatically_declare_parameters_from_overrides, enable_logger_service=enable_logger_service)
        self.get_logger().info("Anti-Glass filter was started!")

        self.declare_parameter("scan_len", 720)
        self.declare_parameter("max_chng", 10000.0) # mm
        self.declare_parameter("stdv_thresh", 175)
        self.declare_parameter("smooth_win", 15) #измеряется в индексах или точках
        self.declare_parameter("min_points", 3)
        self.declare_parameter("min_amp", 2.5)
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("new_scan_topic", "/new_scan")
        self.declare_parameter("min_var_ratio", 0.8)
        self.declare_parameter("frame_id", "laser")

        self.SCAN_LEN = self.get_parameter("scan_len").value
        self.MAX_CHNG = self.get_parameter("max_chng").value
        self.STDV_THRESH = self.get_parameter("stdv_thresh").value
        self.SMOOTH_WIN = self.get_parameter("smooth_win").value
        self.MIN_POINTS = self.get_parameter("min_points").value
        self.SCAN_TOPIC = self.get_parameter("scan_topic").value
        self.NEW_SCAN_TOPIC = self.get_parameter("new_scan_topic").value
        self.MIN_VAR_RATIO = self.get_parameter("min_var_ratio").value
        self.FRAMEID = self.get_parameter("frame_id").value
        self.MIN_AMP = self.get_parameter("min_amp").value

        self.glass_filter = GlassFilter(self.SCAN_LEN)
        self.glass_filter.set_xyt(0, 0, 0) # костыль, без одометрии
        self.get_logger().info(f"Init with these params: scan_len {self.SCAN_LEN} max_chng: {self.MAX_CHNG} stdv_thresh: {self.STDV_THRESH} smooth_win: {self.SMOOTH_WIN} min_points: {self.MIN_POINTS} min_amp: {self.MIN_AMP} scan_topic: {self.SCAN_TOPIC}")

        self.laser_sub = self.create_subscription(msg_type=LaserScan, topic=self.SCAN_TOPIC, qos_profile=10, callback=self.scan_callback)
        self.laser_pub = self.create_publisher(msg_type=LaserScan, topic=self.NEW_SCAN_TOPIC, qos_profile=10)

    def process_scan(self, scan, intens):
        pcs, _, _ = self.glass_filter.stdv_filter(scan, intens, self.STDV_THRESH)
        if pcs != []:
            for pc in pcs.copy():
                valid, dno_idx = self.glass_filter.valid_filter(pc=pc, max_chng=self.MAX_CHNG, kmediansize=self.SMOOTH_WIN, min_points=self.MIN_POINTS, min_amp=self.MIN_AMP)
                if not valid:
                    pcs.remove(pc)
                    continue
                start = pc.points[0].index
                end = pc.points[-1].index
                _, k, b, coords, var_ratio = self.glass_filter.fitting_filter(scan, [start, dno_idx, end], self.MIN_VAR_RATIO)
                print(f"Fitting: {k} {b} {coords} {var_ratio}")
                if var_ratio < self.MIN_VAR_RATIO:
                    pcs.remove(pc)
                    continue
            
            sc, _ = self.glass_filter.patch(pcs, scan)
            return sc
        return scan

    def scan_callback(self, msg: LaserScan):
        scan = np.asarray(msg.ranges)
        scan = np.where(np.isinf(scan), 0.0, scan) # в ros дропаут это inf а не 0.0
        scan *= 1000
        
        intensivities = np.asarray(msg.intensities)

        new_scan = LaserScan()
        new_scan.header.frame_id = self.FRAMEID
        new_scan.header.stamp = msg.header.stamp 

        ranges_data = self.process_scan(scan, intensivities) / 1000
        new_scan.ranges = ranges_data.tolist()

        new_scan.angle_min = float(msg.angle_min)
        new_scan.angle_max = float(msg.angle_max)
        new_scan.angle_increment = float(msg.angle_increment)

        new_scan.range_min = float(np.min(ranges_data))
        new_scan.range_max = float(np.max(ranges_data))

        new_scan.time_increment = float(msg.time_increment)
        new_scan.scan_time = float(msg.scan_time)
        new_scan.intensities = intensivities.tolist()
        self.laser_pub.publish(new_scan)



def main(args=None):
    rclpy.init(args=args)
    filter_node = filter(node_name="anti_glass")
    rclpy.spin(node=filter_node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()