#!/bin/bash

# Build the Docker image
echo "Building Docker image..."
docker build -t gpt-chat-app .

# Run the container (remove old container if exists)
if [ $(docker ps -a -q -f name=gpt-chat-container) ]; then
    echo "Removing existing container..."
    docker rm -f gpt-chat-container
fi

echo "Running Docker container..."
docker run -d --name gpt-chat-container -p 8000:8000 gpt-chat-app

echo "Container is running. Access the app at http://localhost:8000"
