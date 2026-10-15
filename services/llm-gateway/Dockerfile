FROM python:3.12-slim
RUN pip install --no-cache-dir aiohttp==3.12.15
COPY gateway.py /gateway.py
CMD ["python3", "/gateway.py"]
