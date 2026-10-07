#!/usr/bin/env python
"""Convert the camera topics of ROS 2 mcap bags into mp4s in the processed bucket.

For every episode directory under --path (a directory holding one or more
.mcap files, e.g. h2rc/launch_demo/21Aug/press_button_down/<episode>/), each
CompressedImage topic is decoded and piped as JPEG frames into ffmpeg, which
writes H.264 mp4. Results go to s3://processed/<episode path relative to
--root>/<camera>.mp4 plus episode.json (task, day, topics, frame counts, fps,
duration). Episodes whose episode.json exists are skipped.

Needs: pip install mcap mcap-ros2-support imageio-ffmpeg boto3
"""

import argparse
import glob
import json
import os
import subprocess
import tempfile

import boto3
import imageio_ffmpeg
from mcap_ros2.reader import read_ros2_messages
import yaml


def camera_name(topic):
    # /hdas/camera_head/left_raw/image_raw_color/compressed -> camera_head_left
    parts = [p for p in topic.split("/") if p and p not in ("hdas", "compressed", "color", "image_raw", "image_raw_color")]
    return "_".join(p.replace("_raw", "") for p in parts)


def image_topics(episode_dir):
    meta = yaml.safe_load(open(f"{episode_dir}/metadata.yaml"))
    return [t["topic_metadata"]["name"] for t in meta["rosbag2_bagfile_information"]["topics_with_message_count"]
            if t["topic_metadata"]["type"] == "sensor_msgs/msg/CompressedImage"]


def convert_topic(bags, topic, out):
    """Decode one topic from the episode's bags into an mp4; return (frames, fps, seconds)."""
    stamps = []
    ffmpeg = None
    for bag in bags:
        for m in read_ros2_messages(bag, topics=[topic]):
            stamps.append(m.ros_msg.header.stamp.sec + m.ros_msg.header.stamp.nanosec / 1e9)
            if ffmpeg is None:
                # Frame rate isn't known until the end; H.264 with a nominal 30 fps,
                # corrected below by remuxing, keeps this a single pass.
                ffmpeg = subprocess.Popen(
                    [imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "image2pipe", "-c:v", "mjpeg",
                     "-r", "30", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                     "-movflags", "+faststart", out + ".tmp.mp4"],
                    stdin=subprocess.PIPE)
            ffmpeg.stdin.write(bytes(m.ros_msg.data))
    if ffmpeg is None:
        return 0, 0, 0
    ffmpeg.stdin.close()
    ffmpeg.wait()
    seconds = max(stamps) - min(stamps) if len(stamps) > 1 else 0
    fps = round(len(stamps) / seconds, 2) if seconds else 30
    # Remux with the real frame rate (no re-encode).
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-loglevel", "error", "-y", "-r", str(fps), "-i", out + ".tmp.mp4",
                    "-c", "copy", out], check=True)
    os.remove(out + ".tmp.mp4")
    return len(stamps), fps, round(seconds, 3)


def convert_episode(s3, bucket, dest, root, episode_dir):
    rel = os.path.relpath(episode_dir, root)  # keeps the dataset dir, e.g. h2rc/launch_demo/...
    key_prefix = f"{dest}/{rel}" if dest else rel
    try:
        s3.head_object(Bucket=bucket, Key=f"{key_prefix}/episode.json")
        print(f"skip {rel}: already converted")
        return
    except s3.exceptions.ClientError:
        pass

    bags = sorted(glob.glob(f"{episode_dir}/*.mcap"))
    parts = rel.split("/")
    info = {"episode": parts[-1], "task": parts[-2] if len(parts) > 1 else None, "day": parts[-3] if len(parts) > 2 else None,
            "source": rel, "bags": [os.path.basename(b) for b in bags], "cameras": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for topic in image_topics(episode_dir):
            cam = camera_name(topic)
            out = f"{tmp}/{cam}.mp4"
            frames, fps, seconds = convert_topic(bags, topic, out)
            if frames:
                s3.upload_file(out, bucket, f"{key_prefix}/{cam}.mp4")
                info["cameras"][cam] = {"topic": topic, "frames": frames, "fps": fps, "seconds": seconds}
    s3.put_object(Bucket=bucket, Key=f"{key_prefix}/episode.json", Body=json.dumps(info, indent=1).encode())
    print(f"{rel}: {', '.join(f'{c} {v['frames']}f@{v['fps']}' for c, v in info['cameras'].items())}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--path", required=True, help="glob of episode directories under --root, e.g. 'h2rc/launch_demo/*/*/*'")
    ap.add_argument("--dest", default="", help="extra prefix in the processed bucket (the path relative to --root is kept)")
    ap.add_argument("--bucket", default="processed")
    ap.add_argument("--root", default="/buckets/raw")
    ap.add_argument("--limit", type=int, default=0, help="convert at most this many episodes (0 = all)")
    args = ap.parse_args()

    s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"])
    episodes = sorted(d for d in glob.glob(f"{args.root}/{args.path}") if glob.glob(f"{d}/*.mcap"))
    if args.limit:
        episodes = episodes[: args.limit]
    print(f"{len(episodes)} episodes match {args.path}")
    for d in episodes:
        convert_episode(s3, args.bucket, args.dest.strip("/"), args.root, d)


if __name__ == "__main__":
    main()
