# Rerun web viewer. The rerun-sdk wheel ships the CLI with the web viewer
# built in (pure wheel, works on arm64 and amd64).
FROM python:3.12-slim
RUN pip install --no-cache-dir rerun-sdk==0.38.1 && rerun analytics disable
# --serve-web hosts the viewer (web-viewer-port) and a gRPC server (port) the
# viewer connects to by default; recordings are opened with ?url=<rrd>.
CMD ["rerun", "--serve-web", "--bind", "0.0.0.0", "--web-viewer-port", "9090", "--port", "9876", "--hide-welcome-screen", "--server-memory-limit", "4GB"]
