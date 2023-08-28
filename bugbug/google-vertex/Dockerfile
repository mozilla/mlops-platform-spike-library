# FROM python:3.10.7-slim
FROM google/cloud-sdk:latest
WORKDIR /root/bugbug

ADD ../bugbug /root/bugbug

# Install the required deps
RUN apt-get update && \
		apt-get install -y --no-install-recommends gcc g++ libgomp1 libffi-dev libjemalloc2 zstd patch

# Installs additional packages
RUN pip install -r /root/bugbug/requirements.txt

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python3", "-m", "scripts.trainer", "spambug"]
