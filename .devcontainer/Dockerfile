FROM continuumio/miniconda3:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    glpk-utils \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "your_env_name", "/bin/bash", "-c"]

# Add any other packages or tools you need

# Command to run on container start
CMD ["conda", "run", "--no-capture-output", "-n", "your_env_name", "your_command_here"]


RUN pip install --upgrade pip

RUN mkdir -p /home/
COPY ./misc_analytics.yml /app/misc_analytics.yml
RUN conda config --set channel_priority false
RUN conda config --add create_default_packages conda-ecosystem-user-package-isolation

RUN conda env create --file /app/misc_analytics.yml
