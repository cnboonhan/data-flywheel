#!/usr/bin/env python3
"""Make a worldgen capture bag readable by NuRec's rosbag_to_mapping_data.

Nova Carter's Isaac Sim asset stamps camera_info/images with `<cam>_left_optical`, but its TF tree only has the
optical frames under the name `<cam>_left_rgb` (z forward, x right; checked from the bag). This copies the bag and
adds identity /tf_static transforms `<cam>_{left,right}_rgb -> <cam>_{left,right}_optical`, so the converter can
chain the camera to base_link. Output uses mcap chunk-level zstd (the converter can't read rosbag2 per-message
compression). Run in the worldgen nav container:

    python3 /nurec_stereo/fix_bag.py /runs/<name>/bag /runs/<name>/bag_nurec [front_stereo_camera ...]
"""

import sys

import rosbag2_py
from geometry_msgs.msg import TransformStamped
from rclpy.serialization import serialize_message
from tf2_msgs.msg import TFMessage


def main():
    src, dst = sys.argv[1], sys.argv[2]
    cams = sys.argv[3:] or ["front_stereo_camera"]
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=src, storage_id="mcap"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    writer = rosbag2_py.SequentialWriter()
    writer.open(rosbag2_py.StorageOptions(uri=dst, storage_id="mcap", storage_preset_profile="zstd_fast"),
                rosbag2_py.ConverterOptions("cdr", "cdr"))
    for meta in reader.get_all_topics_and_types():
        writer.create_topic(meta)
    writer.create_topic(rosbag2_py.TopicMetadata(
        0, "/tf_static", "tf2_msgs/msg/TFMessage", "cdr",
        [rosbag2_py.QoS(1).reliable().transient_local()] if hasattr(rosbag2_py, "QoS") else []))

    first, n = True, 0
    while reader.has_next():
        topic, data, stamp = reader.read_next()
        if first:
            msg = TFMessage()
            for cam in cams:
                for side in ("left", "right"):
                    t = TransformStamped()
                    t.header.stamp.sec, t.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
                    t.header.frame_id = f"{cam}_{side}_rgb"
                    t.child_frame_id = f"{cam}_{side}_optical"
                    t.transform.rotation.w = 1.0
                    msg.transforms.append(t)
            writer.write("/tf_static", serialize_message(msg), stamp)
            first = False
        writer.write(topic, data, stamp)
        n += 1
        if n % 50000 == 0:
            print(f"{n} messages", flush=True)
    print(f"done: {n} messages + /tf_static aliases for {cams}")


if __name__ == "__main__":
    main()
