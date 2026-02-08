FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

COPY python/lob_rl/ /app/python/lob_rl/
COPY scripts/train.py /app/scripts/train.py

WORKDIR /app
RUN pip install gymnasium numpy stable-baselines3 sb3-contrib tensorboard

ENV PYTHONPATH=/app/python
ENTRYPOINT ["python", "/app/scripts/train.py"]
