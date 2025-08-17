# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 803108168810.dkr.ecr.us-east-1.amazonaws.com

echo "Pulling Docker image..."
docker pull 803108168810.dkr.ecr.us-east-1.amazonaws.com/youtube_comment_analysis:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=youtube-comment-app)" ]; then
    echo "Stopping existing container..."
    docker stop youtube-comment-app
fi

if [ "$(docker ps -aq -f name=youtube-comment-app)" ]; then
    echo "Removing existing container..."
    docker rm youtube-comment-app
fi

echo "Starting new container..."
docker run -d -p 80:5000 --name youtube-comment-app \
    803108168810.dkr.ecr.us-east-1.amazonaws.com/youtube_comment_analysis:latest

echo "Container started successfully."
