#!/bin/bash

# Check if an argument was passed
if [ -z "$1" ]; then
  echo "Usage: ./run-container.sh [jupyter|code-server]"
fi

# Set container name and image
CONTAINER_NAME="cpe727-regularization-cnt"
IMAGE_NAME="cpe727-regularization"

# Check if the container is already running
if [ "$(docker ps -q -f name=$CONTAINER_NAME)" ]; then
  echo "Stopping existing container..."
  docker stop $CONTAINER_NAME
  docker rm $CONTAINER_NAME
fi

# Run the container and choose between Jupyter or code-server
if [ "$1" == "jupyter" ]; then
  echo "Starting Jupyter Lab..."
  docker run -d -p 10001:8888 -v $(pwd):/home/code --name $CONTAINER_NAME $IMAGE_NAME bash -c "cd /home/code && jupyter lab --ip=0.0.0.0 --allow-root"
  
  sleep 5

  # Retrieve the Jupyter token from the logs
  JUPYTER_TOKEN=$(docker logs $CONTAINER_NAME 2>&1 | grep -oP '(?<=token=)[a-zA-Z0-9]+')

  echo "Jupyter Lab is running at http://localhost:10001"
  echo "Token: $JUPYTER_TOKEN"
elif [ "$1" == "code-server" ]; then
  echo "Starting code-server..."
  docker run -d -p 10000:10000 -v $(pwd):/home/code --name $CONTAINER_NAME $IMAGE_NAME code-server --port 10000

  sleep 5
  # Retrieve the password from code-server's config file inside the container
  PASSWORD=$(docker exec $CONTAINER_NAME cat /root/.config/code-server/config.yaml | grep "password:" | awk '{print $2}')

  echo "code-server is running at http://localhost:10000"
  echo "Password: $PASSWORD"
else
  echo "Invalid option. Use 'jupyter' or 'code-server'."
fi
