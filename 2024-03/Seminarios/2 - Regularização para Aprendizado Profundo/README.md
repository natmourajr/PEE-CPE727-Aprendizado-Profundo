# Seminar on Regularization for Deep Neural Networks

The seminar is based on Chapter 7 of [The Deep Learning book](https://www.deeplearningbook.org/)

This repository provides a Docker-based setup with a PyTorch environment, equipped for remote development with both Jupyter Lab and code-server (VS Code in the browser).

## Table of Contents
1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Building the Docker Image](#building-the-docker-image)
4. [Running the Docker Container Locally](#running-the-docker-container-locally)
5. [Deploying on SLURM with Singularity](#deploying-on-slurm-with-singularity)
6. [Port Forwarding for Remote Access](#port-forwarding-for-remote-access)

## Features
- **Jupyter Lab**: Web-based, interactive development for notebooks, code, and data exploration.
- **code-server**: VS Code in the browser, enabling remote development similar to a local VS Code environment.

## Prerequisites
- **Docker**: Install Docker on your machine ([Get Docker](https://docs.docker.com/get-docker/)).
- **Singularity (for SLURM clusters)**: For deployments on cluster environments using SLURM.

## Building the Docker Image
1. Clone this repository.
2. Build the Docker image by running the following command in the directory containing your `Dockerfile`:

   ```bash
   docker build -t cpe727-regularization .
   ```

## Running the Docker Container Locally
After building the image, you can run the container with Jupyter Lab or code-server:

### Start Jupyter Lab
```bash
source run-container.sh jupyter
```

### Start code-server
```bash
source run-container.sh code-server
```

### Troubleshooting
If you encounter an error about the container already running or registered, stop and remove the container:

```bash
docker stop cpe727-regularization-cnt
docker rm cpe727-regularization-cnt
```

## Deploying on SLURM with Singularity

### Step 1: Push Docker Image to Docker Hub
1. Sign in to Docker Hub:
   ```bash
   docker login -u <username>
   ```
2. Build and tag your image, then push it to Docker Hub:
   ```bash
   docker build -t <username>/cpe727-regularization .
   docker push <username>/cpe727-regularization
   ```

### Step 2: Pull Image on SLURM Cluster
On the cluster, alocate a machine and clone this repository and pull the image as a `.sif` file:

```bash
singularity pull --disable-cache cpe727-regularization.sif docker://<username>/cpe727-regularization
```

## Port Forwarding for Remote Access
To access Jupyter Lab or code-server remotely, you’ll need to set up port forwarding:

1. **Local to Login Node (SSH Tunnel)**: Forward from your machine to the login node.
2. **Login Node to Compute Node**: Forward from the login node to the allocated compute node.

Detailed instructions on port forwarding can be found [here](https://www.ssh.com/ssh/tunneling/example).
