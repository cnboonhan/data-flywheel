#!/usr/bin/env python
"""Convert ROS 2 mcap episodes (h2rc) into the canonical episode layout.

Each episode directory (holding *.mcap and metadata.yaml) becomes
processed/episodes/<dataset>/<path relative to --root>/ with one mp4 per
CompressedImage topic (JPEG frames piped through ffmpeg, H.264) and a
signals.parquet from every JointState topic (position, velocity, effort per
joint) plus IMU and wrench topics.

Needs: pip install mcap mcap-ros2-support imageio-ffmpeg boto3 pyyaml pyarrow
"""

import argparse
import glob
import os
import subprocess

import imageio_ffmpeg
import yaml
from mcap_ros2.reader import read_ros2_messages

import episodes as ep


def camera_name(topic):
    parts = [p for p in topic.split("/") if p and p not in ("hdas", "compressed", "color", "image_raw", "image_raw_color")]
    return "_".join(p.replace("_raw", "") for p in parts)


def topics(episode_dir):
    meta = yaml.safe_load(open(f"{episode_dir}/metadata.yaml"))
    return {t["topic_metadata"]["name"]: t["topic_metadata"]["type"] for t in meta["rosbag2_bagfile_information"]["topics_with_message_count"]}


def stamp(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9


def convert_video(bags, topic, out):
    stamps, ffmpeg = [], None
    for bag in bags:
        for m in read_ros2_messages(bag, topics=[topic]):
            stamps.append(stamp(m.ros_msg))
            if ffmpeg is None:
                ffmpeg = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "image2pipe", "-c:v", "mjpeg", "-r", "30", "-i", "-",
                                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out + ".tmp.mp4"],
                                          stdin=subprocess.PIPE)
            ffmpeg.stdin.write(bytes(m.ros_msg.data))
    if ffmpeg is None:
        return None
    ffmpeg.stdin.close(); ffmpeg.wait()
    seconds = max(stamps) - min(stamps) if len(stamps) > 1 else 0
    fps = round(len(stamps) / seconds, 2) if seconds else 30
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-r", str(fps), "-i", out + ".tmp.mp4", "-c", "copy", out], check=True)
    os.remove(out + ".tmp.mp4")
    return {"frames": len(stamps), "fps": fps, "t0": min(stamps), "seconds": round(seconds, 3)}


def signal_rows(bags, kinds, t0):
    rows = []
    for topic, typ in kinds.items():
        group = topic.strip("/").replace("/", ".")
        for bag in bags:
            for m in read_ros2_messages(bag, topics=[topic]):
                msg, t = m.ros_msg, None
                try:
                    t = stamp(msg) - t0
                except AttributeError:
                    t = m.log_time_ns / 1e9 - t0
                if typ == "sensor_msgs/msg/JointState":
                    for field in ("position", "velocity", "effort"):
                        vals = getattr(msg, field, None) or []
                        rows += [(t, f"{group}.{field}", i, float(v)) for i, v in enumerate(vals)]
                elif typ == "sensor_msgs/msg/Imu":
                    rows += [(t, f"{group}.angular_velocity", i, float(v)) for i, v in enumerate((msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z))]
                    rows += [(t, f"{group}.linear_acceleration", i, float(v)) for i, v in enumerate((msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z))]
                elif typ == "geometry_msgs/msg/WrenchStamped":
                    w = msg.wrench
                    rows += [(t, f"{group}.force", i, float(v)) for i, v in enumerate((w.force.x, w.force.y, w.force.z))]
                    rows += [(t, f"{group}.torque", i, float(v)) for i, v in enumerate((w.torque.x, w.torque.y, w.torque.z))]
    return rows


def convert_episode(client, dataset, root, episode_dir, tmp):
    eid = os.path.relpath(episode_dir, root)
    if ep.exists(client, dataset, eid):
        return False
    bags = sorted(glob.glob(f"{episode_dir}/*.mcap"))
    kinds = topics(episode_dir)
    cams, t0 = {}, None
    for topic, typ in kinds.items():
        if typ != "sensor_msgs/msg/CompressedImage":
            continue
        cam = camera_name(topic)
        out = f"{tmp}/{cam}.mp4"
        info = convert_video(bags, topic, out)
        if info:
            ep.upload_file(client, dataset, eid, f"{cam}.mp4", out)
            t0 = info["t0"] if t0 is None else min(t0, info["t0"])
            cams[cam] = {"topic": topic, **{k: v for k, v in info.items() if k != "t0"}}
    signals = {k: v for k, v in kinds.items() if v in ("sensor_msgs/msg/JointState", "sensor_msgs/msg/Imu", "geometry_msgs/msg/WrenchStamped") and not k.startswith("/debug")}
    n = ep.write_signals(client, dataset, eid, signal_rows(bags, signals, t0 or 0.0))
    parts = eid.split("/")
    ep.write_json(client, dataset, eid, {
        "dataset": dataset, "episode_id": eid, "source": os.path.relpath(episode_dir, "/buckets"), "format": "ros2-mcap", "robot": "r1",
        "task": parts[-2] if len(parts) > 1 else None, "tasks": [parts[-2]] if len(parts) > 1 else [], "day": parts[-3] if len(parts) > 2 else None,
        "fps": max((c["fps"] for c in cams.values()), default=None), "frames": max((c["frames"] for c in cams.values()), default=0),
        "duration_s": max((c["seconds"] for c in cams.values()), default=0), "cameras": cams, "signal_rows": n, "bags": [os.path.basename(b) for b in bags]})
    print(f"{eid}: {len(cams)} cameras, {n} signal rows")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="h2rc")
    ap.add_argument("--path", required=True, help="glob of episode directories under --root, e.g. 'h2rc/launch_demo/*/*/*'")
    ap.add_argument("--root", default="/buckets/raw")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    client = ep.s3()
    dirs = sorted(d for d in glob.glob(f"{args.root}/{args.path}") if glob.glob(f"{d}/*.mcap"))
    if args.limit:
        dirs = dirs[: args.limit]
    print(f"{len(dirs)} episodes match {args.path}")
    root = f"{args.root}/{args.dataset}"
    with ep.tmpdir() as tmp:
        done = sum(convert_episode(client, args.dataset, root, d, tmp) for d in dirs)
    print(f"{done} new episodes -> s3://processed/episodes/{args.dataset}/")


if __name__ == "__main__":
    main()
