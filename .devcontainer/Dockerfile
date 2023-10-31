FROM continuumio/miniconda3:latest

WORKDIR /app

#docker build -t misc-analytics:latest -f ./.devcontainer/Dockerfile .

RUN apt-get update && apt-get install -y \
    glpk-utils \
    git \
    vim \
    build-essential \
    curl \
    libgl1-mesa-glx \
    tmux \
    openssl \
    libssl-dev \
    gnupg \
    apt-transport-https \
    libterm-readline-gnu-perl \
    gcc \
    libc-dev \
    g++ \
    unzip \
    && rm -rf /var/lib/apt/lists/*


COPY ./env_files/misc_analytics_no_versions.yml /app/misc_analytics_no_versions.yml
RUN conda config --set channel_priority false
RUN conda config --add create_default_packages conda-ecosystem-user-package-isolation

RUN conda install mamba -c conda-forge
RUN mamba env create --file /app/misc_analytics_no_versions.yml

ENTRYPOINT ["/bin/bash", "-c", "sleep infinity"]