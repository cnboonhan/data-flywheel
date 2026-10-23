"""A ROS 2 bag (.mcap) as a Rerun recording small enough for the web viewer: build(src, out, progress).

The browser holds a recording in its tab's ~4 GB, and a robot's bag is mostly camera frames as JPEG: ~4 MB/s for four
640x360 cameras, so a 7-minute bag is ~1.5 GB and the web viewer only partly loads it. Here each camera becomes an
H.264 video (ffmpeg; ~7x smaller than its JPEGs) and the rest of what the bag holds is kept as plots: JointState
positions, velocities and efforts per joint, Imu, WrenchStamped, TwistStamped; String topics as a text log. Every
other message type is listed under `skipped`. All on a `time` timeline with the bag's own timestamps (log time).

One pass over the bag, decoding only the topics it keeps: ~20k messages/s, so ~40 s for a 7-minute bag, while the
cameras encode alongside (~15x real time each). progress(read, total) is called as messages are read.
"""
import os
import re
import subprocess
import tempfile
from collections import defaultdict

import numpy as np
import rerun as rr
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

IMAGE = "sensor_msgs/msg/CompressedImage"
FPS = 30   # the cameras' nominal rate: the video's own clock; each frame is placed at its message's time anyway
PROGRESS_EVERY = 5000   # messages between progress calls: ~4 per second at the read rate above


def entity(topic):
    """A topic as an entity path: /hdas/camera_head/left_raw/... -> hdas/camera_head/left_raw/..."""
    return topic.strip("/")


def joint_names(names, n):
    """A JointState's names, one per value: some publishers send them as one comma-joined string (/calib/joint_bias);
    without names, or with a count that doesn't match, the values' indices. Each safe as one entity-path part."""
    names = list(names)
    if len(names) == 1 and n > 1:
        names = [s.strip() for s in names[0].split(",")]
    if len(names) != n:
        names = [str(i) for i in range(n)]
    return [re.sub(r"[^A-Za-z0-9_.-]+", "_", s) or str(i) for i, s in enumerate(names)]


def vec(prefix, obj, fields, out, t):
    for f in fields:
        out[f"{prefix}/{f}"].append((t, float(getattr(obj, f))))


def build(src, out, progress=lambda read, total: None):
    with open(src, "rb") as f, tempfile.TemporaryDirectory(dir=os.path.dirname(out)) as tmp:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        summary = reader.get_summary()
        total = summary.statistics.message_count
        schema = {cid: summary.schemas[c.schema_id].name if c.schema_id in summary.schemas else "" for cid, c in summary.channels.items()}
        series = defaultdict(list)           # entity path -> [(time ns, value)]
        texts = defaultdict(list)            # entity path -> [(time ns, text)]
        video, frames = {}, defaultdict(list)  # topic -> (ffmpeg, mp4 path), topic -> [time ns]
        skipped = defaultdict(int)
        decoders = {}
        read = 0
        for sch, ch, msg in reader.iter_messages():
            read += 1
            if read % PROGRESS_EVERY == 0:
                progress(read, total)
            kind = schema.get(ch.id, "")
            if kind not in (IMAGE, "sensor_msgs/msg/JointState", "sensor_msgs/msg/Imu", "geometry_msgs/msg/WrenchStamped",
                            "geometry_msgs/msg/TwistStamped", "std_msgs/msg/String"):
                skipped[f"{ch.topic} ({kind})"] += 1
                continue
            dec = decoders.get(ch.id)
            if dec is None:
                dec = decoders[ch.id] = DecoderFactory().decoder_for(ch.message_encoding, sch)
            m = dec(msg.data)
            t, e = msg.log_time, entity(ch.topic)
            if kind == IMAGE:
                if ch.topic not in video:
                    mp4 = os.path.join(tmp, re.sub(r"[^A-Za-z0-9]+", "_", e) + ".mp4")
                    # crf 23, keyframe every 2 s: viewer-grade and small, as the LeRobot episodes'; untuned
                    video[ch.topic] = (subprocess.Popen(
                        ["ffmpeg", "-v", "error", "-y", "-f", "image2pipe", "-c:v", "mjpeg", "-framerate", str(FPS), "-i", "-",
                         "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-g", str(2 * FPS), mp4],
                        stdin=subprocess.PIPE), mp4)
                video[ch.topic][0].stdin.write(bytes(m.data))
                frames[ch.topic].append(t)
            elif kind == "sensor_msgs/msg/JointState":
                for field in ("position", "velocity", "effort"):
                    vals = getattr(m, field)
                    for n, v in zip(joint_names(m.name, len(vals)), vals):
                        series[f"{e}/{field}/{n}"].append((t, float(v)))
            elif kind == "sensor_msgs/msg/Imu":
                vec(f"{e}/orientation", m.orientation, "xyzw", series, t)
                vec(f"{e}/angular_velocity", m.angular_velocity, "xyz", series, t)
                vec(f"{e}/linear_acceleration", m.linear_acceleration, "xyz", series, t)
            elif kind == "geometry_msgs/msg/WrenchStamped":
                vec(f"{e}/force", m.wrench.force, "xyz", series, t)
                vec(f"{e}/torque", m.wrench.torque, "xyz", series, t)
            elif kind == "geometry_msgs/msg/TwistStamped":
                vec(f"{e}/linear", m.twist.linear, "xyz", series, t)
                vec(f"{e}/angular", m.twist.angular, "xyz", series, t)
            else:
                texts[e].append((t, str(m.data)))
        progress(total, total)
        for ff, _ in video.values():
            ff.stdin.close()
        for topic, (ff, _) in video.items():
            if ff.wait():
                raise RuntimeError(f"ffmpeg failed on {topic}")

        rec = rr.RecordingStream("bag", recording_id=os.path.abspath(src))
        rec.save(out)
        for path, pts in series.items():
            a = np.asarray(pts)
            rec.send_columns(path, indexes=[rr.TimeColumn("time", timestamp=a[:, 0] * 1e-9)],
                             columns=rr.Scalars.columns(scalars=a[:, 1]))
        for path, lines in texts.items():
            for t, s in lines:
                rec.set_time("time", timestamp=t * 1e-9)
                rec.log(path, rr.TextLog(s))
        for topic, (_, mp4) in video.items():
            asset = rr.AssetVideo(path=mp4)
            e = entity(topic)
            rec.log(e, asset, static=True)
            ts = asset.read_frame_timestamps_nanos()
            n = min(len(ts), len(frames[topic]))   # one video frame per message: ffmpeg keeps every one at a fixed rate
            rec.send_columns(e, indexes=[rr.TimeColumn("time", timestamp=np.asarray(frames[topic][:n]) * 1e-9)],
                             columns=rr.VideoFrameReference.columns_nanos(ts[:n]))
        if skipped:
            rec.log("skipped", rr.TextDocument("Topics not shown here (messages):\n" + "\n".join(
                f"{k}: {v}" for k, v in sorted(skipped.items()))), static=True)
        rec.flush()
        rec.disconnect()
