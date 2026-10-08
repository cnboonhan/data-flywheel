# FiftyOne for the App and the sync-fiftyone-raw job, so both run one version.
# Adds what VFF_MULTIMODAL needs (protobuf for the multimodal routes, pyarrow
# for LeRobot sources) and h5py for reading xspark episodes. The stock image
# ships none of them.
FROM voxel51/fiftyone:latest
RUN /opt/.fiftyone-venv/bin/pip install --no-cache-dir protobuf pyarrow h5py
