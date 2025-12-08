#!/usr/bin/env python3
import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2


class PointCloudDownsampler(Node):
    def __init__(self):
        super().__init__("pointcloud_downsampler")

        self.declare_parameter("input_topic", "/input_cloud")
        self.declare_parameter("output_topic", "/downsampled_cloud")
        self.declare_parameter("leaf_size", 0.05)  # meters

        self.input_topic = (
            self.get_parameter("input_topic").get_parameter_value().string_value
        )
        self.output_topic = (
            self.get_parameter("output_topic").get_parameter_value().string_value
        )
        self.leaf_size = (
            self.get_parameter("leaf_size").get_parameter_value().double_value
        )

        # Subscriber & publisher
        self.sub = self.create_subscription(
            PointCloud2,
            self.input_topic,
            self.cloud_callback,
            10,  # queue size
        )
        self.pub = self.create_publisher(
            PointCloud2,
            self.output_topic,
            10,
        )

        self.get_logger().info(
            f"PointCloudDownsampler initialized.\n"
            f"  Subscribing to: {self.input_topic}\n"
            f"  Publishing to:  {self.output_topic}\n"
            f"  leaf_size:      {self.leaf_size:.3f} m"
        )

    def cloud_callback(self, cloud_msg: PointCloud2):
        # Read all points (skip NaNs)
        points_iter = pc2.read_points(cloud_msg, field_names=None, skip_nans=True)
        leaf = float(self.leaf_size)

        # voxel index -> representative point
        voxels = {}

        for p in points_iter:
            # Assume first 3 fields are x,y,z
            x, y, z = p[0], p[1], p[2]

            ix = int(math.floor(x / leaf))
            iy = int(math.floor(y / leaf))
            iz = int(math.floor(z / leaf))
            key = (ix, iy, iz)

            # Keep first point per voxel (simple, fast)
            if key not in voxels:
                voxels[key] = p

        filtered_points = list(voxels.values())

        # Create new PointCloud2, reusing original fields (keeps rgb/intensity/etc.)
        header = cloud_msg.header
        filtered_msg = pc2.create_cloud(header, cloud_msg.fields, filtered_points)

        self.pub.publish(filtered_msg)
        self.get_logger().debug(
            f"Downsampled from {cloud_msg.width * cloud_msg.height} "
            f"to {len(filtered_points)} points."
        )


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudDownsampler()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

