FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update -qq && apt-get install -y -qq openssh-server rsync && \
    mkdir -p /run/sshd && \
    echo 'PermitRootLogin yes' >> /etc/ssh/sshd_config && \
    rm -rf /var/lib/apt/lists/*

COPY python/lob_rl/ /app/python/lob_rl/
COPY scripts/train.py /app/scripts/train.py
COPY scripts/start.sh /app/scripts/start.sh

WORKDIR /app
RUN pip install gymnasium numpy stable-baselines3 sb3-contrib tensorboard

ENV PYTHONPATH=/app/python
ENTRYPOINT ["/app/scripts/start.sh"]
