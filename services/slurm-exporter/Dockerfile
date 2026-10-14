# The Slurm client comes from the host (/cm/local/apps/slurm, mounted by docker-compose.yml); the image adds only what
# it links against beyond glibc (munge, readline) and Python for the exporter.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends libmunge2 libreadline8t64 python3 \
 && rm -rf /var/lib/apt/lists/*
COPY exporter.py /exporter.py
CMD ["python3", "/exporter.py"]
