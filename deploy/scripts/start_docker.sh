# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

: "${AWS_ACCOUNT_ID:?AWS_ACCOUNT_ID environment variable is not set}"
: "${AWS_DEFAULT_REGION:=us-east-1}"
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_DEFAULT_REGION}.amazonaws.com"
IMAGE_URI="${ECR_REGISTRY}/youtube_comment_analysis:latest"

echo "Logging in to ECR..."
aws ecr get-login-password --region "${AWS_DEFAULT_REGION}" | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

echo "Pulling Docker image..."
docker pull "${IMAGE_URI}"

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
    -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
    -e MODEL_NAME="${MODEL_NAME:-yt_chrome_plugin_model}" \
    -e MODEL_VERSION="${MODEL_VERSION:-Production}" \
    "${IMAGE_URI}"

echo "Container started successfully."
