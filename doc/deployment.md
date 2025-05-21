# Deployment Guide - News Analysis Application

## Table of Contents
- [Deployment Overview](#deployment-overview)
- [Local Development Deployment](#local-development-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Platform Deployment](#cloud-platform-deployment)
- [Production Deployment](#production-deployment)
- [CI/CD Pipeline](#cicd-pipeline)
- [Monitoring and Maintenance](#monitoring-and-maintenance)
- [Scaling and Load Balancing](#scaling-and-load-balancing)

## Deployment Overview

The News Analysis application can be deployed in various environments, from local development to cloud-based production systems. This guide covers multiple deployment strategies to suit different needs.

### Deployment Options
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Local Dev       │    │ Docker          │    │ Cloud Platform  │
│                 │    │                 │    │                 │
│ - Development   │    │ - Containerized │    │ - AWS/GCP/Azure │
│ - Testing       │    │ - Portable      │    │ - Managed       │
│ - Debugging     │    │ - Consistent    │    │ - Scalable      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Prerequisites for All Deployments
- Python 3.8+
- Required system dependencies (see [Installation Guide](installation.md))
- API keys configured (OpenAI, Pinecone)
- Sufficient compute resources

## Local Development Deployment

### Quick Start (Development Mode)
```bash
# Clone repository
git clone https://github.com/your-repo/newsAnalysis.git
cd newsAnalysis

# Set up virtual environment
python -m venv news_analysis_env
source news_analysis_env/bin/activate  # Linux/macOS
# news_analysis_env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run application
streamlit run main.py
```

### Development Configuration
Create `config/development.yaml`:
```yaml
environment: development
debug: true
log_level: DEBUG

server:
  host: localhost
  port: 8501
  reload: true

features:
  enable_caching: false
  enable_metrics: false
  verbose_logging: true

models:
  cache_models: true
  gpu_enabled: auto
```

### Development Scripts
Create `scripts/dev.sh`:
```bash
#!/bin/bash
# Development startup script

echo "Starting News Analysis Application (Development Mode)"

# Activate virtual environment
source news_analysis_env/bin/activate

# Set development environment
export ENVIRONMENT=development
export DEBUG=True

# Start application with auto-reload
streamlit run main.py --server.runOnSave true
```

## Docker Deployment

### Single Container Deployment

#### 1. Create Dockerfile
```dockerfile
# Dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    portaudio19-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### 2. Build and Run Container
```bash
# Build Docker image
docker build -t news-analysis:latest .

# Run container
docker run -d \
    --name news-analysis \
    -p 8501:8501 \
    -v $(pwd)/.env:/app/.env:ro \
    -v $(pwd)/models:/app/models:ro \
    --restart unless-stopped \
    news-analysis:latest
```

### Multi-Container Deployment (Docker Compose)

#### 1. Create docker-compose.yml
```yaml
version: '3.8'

services:
  news-analysis:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: news-analysis-app
    ports:
      - "8501:8501"
    environment:
      - ENVIRONMENT=production
      - REDIS_URL=redis://redis:6379/0
    volumes:
      - ./.env:/app/.env:ro
      - ./models:/app/models:ro
      - news_analysis_temp:/app/temp
    depends_on:
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8501/_stcore/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis:
    image: redis:7-alpine
    container_name: news-analysis-redis
    volumes:
      - redis_data:/data
    restart: unless-stopped
    command: redis-server --appendonly yes
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  nginx:
    image: nginx:alpine
    container_name: news-analysis-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - news-analysis
    restart: unless-stopped

volumes:
  redis_data:
  news_analysis_temp:
```

#### 2. Nginx Configuration
Create `nginx/nginx.conf`:
```nginx
events {
    worker_connections 1024;
}

http {
    upstream news_analysis {
        server news-analysis:8501;
    }

    server {
        listen 80;
        server_name your-domain.com;
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl;
        server_name your-domain.com;

        ssl_certificate /etc/nginx/ssl/cert.pem;
        ssl_certificate_key /etc/nginx/ssl/key.pem;

        location / {
            proxy_pass http://news_analysis;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            # WebSocket support for Streamlit
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";

            # Timeout settings
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }
    }
}
```

#### 3. Deploy with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Update application
docker-compose pull
docker-compose up -d --build
```

## Cloud Platform Deployment

### AWS Deployment (ECS + Fargate)

#### 1. Task Definition
Create `aws/task-definition.json`:
```json
{
  "family": "news-analysis",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "news-analysis",
      "image": "your-account.dkr.ecr.region.amazonaws.com/news-analysis:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "essential": true,
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:openai-api-key"
        },
        {
          "name": "PINECONE_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:pinecone-api-key"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/news-analysis",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8501/_stcore/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3
      }
    }
  ]
}
```

#### 2. Service Definition
```yaml
# aws/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: news-analysis-service
spec:
  desiredCount: 2
  launchType: FARGATE
  networkConfiguration:
    awsvpcConfiguration:
      subnets:
        - subnet-12345
        - subnet-67890
      securityGroups:
        - sg-abcdef
      assignPublicIp: ENABLED
  loadBalancers:
    - targetGroupArn: arn:aws:elasticloadbalancing:region:account:targetgroup/news-analysis
      containerName: news-analysis
      containerPort: 8501
```

#### 3. Deployment Script
```bash
#!/bin/bash
# aws/deploy.sh

# Build and push Docker image
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-west-2.amazonaws.com

docker build -t news-analysis .
docker tag news-analysis:latest your-account.dkr.ecr.us-west-2.amazonaws.com/news-analysis:latest
docker push your-account.dkr.ecr.us-west-2.amazonaws.com/news-analysis:latest

# Update ECS service
aws ecs register-task-definition --cli-input-json file://aws/task-definition.json
aws ecs update-service --cluster news-analysis-cluster --service news-analysis-service --task-definition news-analysis
```

### Google Cloud Platform (Cloud Run)

#### 1. Cloud Build Configuration
Create `cloudbuild.yaml`:
```yaml
steps:
  # Build the container image
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/news-analysis:$COMMIT_SHA', '.']

  # Push the container image to Container Registry
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/news-analysis:$COMMIT_SHA']

  # Deploy container image to Cloud Run
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: gcloud
    args:
      - 'run'
      - 'deploy'
      - 'news-analysis'
      - '--image'
      - 'gcr.io/$PROJECT_ID/news-analysis:$COMMIT_SHA'
      - '--region'
      - 'us-central1'
      - '--platform'
      - 'managed'
      - '--allow-unauthenticated'
      - '--memory'
      - '2Gi'
      - '--cpu'
      - '2'
      - '--timeout'
      - '300'
      - '--max-instances'
      - '10'

images:
  - 'gcr.io/$PROJECT_ID/news-analysis:$COMMIT_SHA'
```

#### 2. Deploy to Cloud Run
```bash
# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com

# Set environment variables
gcloud run services update news-analysis \
  --set-env-vars OPENAI_API_KEY=$OPENAI_API_KEY,PINECONE_API_KEY=$PINECONE_API_KEY \
  --region us-central1

# Deploy using Cloud Build
gcloud builds submit --config cloudbuild.yaml
```

### Azure Container Instances

#### 1. Azure Resource Manager Template
Create `azure/template.json`:
```json
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "containerName": {
      "type": "string",
      "defaultValue": "news-analysis"
    },
    "imageRegistry": {
      "type": "string"
    },
    "imageName": {
      "type": "string",
      "defaultValue": "news-analysis:latest"
    }
  },
  "resources": [
    {
      "type": "Microsoft.ContainerInstance/containerGroups",
      "apiVersion": "2019-12-01",
      "name": "[parameters('containerName')]",
      "location": "[resourceGroup().location]",
      "properties": {
        "containers": [
          {
            "name": "[parameters('containerName')]",
            "properties": {
              "image": "[concat(parameters('imageRegistry'), '/', parameters('imageName'))]",
              "ports": [
                {
                  "port": 8501,
                  "protocol": "TCP"
                }
              ],
              "resources": {
                "requests": {
                  "cpu": 1,
                  "memoryInGb": 2
                }
              },
              "environmentVariables": [
                {
                  "name": "ENVIRONMENT",
                  "value": "production"
                }
              ]
            }
          }
        ],
        "osType": "Linux",
        "ipAddress": {
          "type": "Public",
          "ports": [
            {
              "port": 8501,
              "protocol": "TCP"
            }
          ],
          "dnsNameLabel": "[parameters('containerName')]"
        }
      }
    }
  ]
}
```

## Production Deployment

### Production Checklist

#### Security
- [ ] HTTPS configured with valid SSL certificates
- [ ] API keys stored in secure secret management
- [ ] Input validation and sanitization enabled
- [ ] Rate limiting configured
- [ ] CORS properly configured
- [ ] Security headers set

#### Performance
- [ ] Caching enabled (Redis/Memcached)
- [ ] CDN configured for static assets
- [ ] Database connection pooling
- [ ] Resource limits set
- [ ] Auto-scaling configured
- [ ] Load balancer health checks

#### Monitoring
- [ ] Application logging configured
- [ ] Error tracking (Sentry/Rollbar)
- [ ] Performance monitoring (APM)
- [ ] Infrastructure monitoring
- [ ] Alerting rules set up
- [ ] Backup strategy implemented

#### Reliability
- [ ] Database backups automated
- [ ] Zero-downtime deployment process
- [ ] Rollback procedures documented
- [ ] Disaster recovery plan
- [ ] Health checks configured
- [ ] Circuit breakers implemented

### Production Environment Variables
```env
# Production settings
ENVIRONMENT=production
DEBUG=False
LOG_LEVEL=WARNING

# Security
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=your-domain.com,www.your-domain.com
SECURE_SSL_REDIRECT=True
SECURE_COOKIES=True

# Database
DATABASE_URL=postgresql://user:pass@host:5432/dbname
REDIS_URL=redis://redis-host:6379/0

# External Services
OPENAI_API_KEY=prod-openai-key
PINECONE_API_KEY=prod-pinecone-key
PINECONE_ENVIRONMENT=production

# Monitoring
SENTRY_DSN=your-sentry-dsn
NEW_RELIC_LICENSE_KEY=your-newrelic-key

# Performance
CACHE_TIMEOUT=3600
MAX_WORKERS=4
REQUEST_TIMEOUT=30
```

## CI/CD Pipeline

### GitHub Actions Workflow
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy News Analysis

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3

    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        python -m spacy download en_core_web_sm

    - name: Run tests
      run: |
        python -m pytest tests/ -v

    - name: Run linting
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics

  build-and-deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'

    steps:
    - uses: actions/checkout@v3

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest

    - name: Deploy to production
      run: |
        # Add your deployment commands here
        # e.g., kubectl apply -f k8s/
        # or aws ecs update-service --cluster prod --service news-analysis
        echo "Deploying to production..."
```

## Monitoring and Maintenance

### Application Monitoring

#### 1. Health Check Endpoint
Add to your application:
```python
# health.py
@app.route('/health')
def health_check():
    """Health check endpoint for load balancers."""
    try:
        # Test critical components
        test_database_connection()
        test_model_loading()
        test_external_apis()

        return {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'version': get_app_version()
        }, 200
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }, 500
```

#### 2. Metrics Collection
```python
# metrics.py
import time
from functools import wraps

def track_execution_time(operation_name):
    """Decorator to track function execution time."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Send metrics to monitoring service
                send_metric(f'{operation_name}.success', execution_time)
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                send_metric(f'{operation_name}.error', execution_time)
                raise
        return wrapper
    return decorator

@track_execution_time('ocr_processing')
def extract_text(image):
    # Your OCR code here
    pass
```

### Log Management
```yaml
# docker-compose.logging.yml
version: '3.8'
services:
  news-analysis:
    # ... other config
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Add centralized logging
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:7.15.0
    environment:
      - discovery.type=single-node
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  kibana:
    image: docker.elastic.co/kibana/kibana:7.15.0
    ports:
      - "5601:5601"
    depends_on:
      - elasticsearch

volumes:
  elasticsearch_data:
```

### Backup Strategy
```bash
#!/bin/bash
# scripts/backup.sh

# Database backup
pg_dump $DATABASE_URL > backups/db_$(date +%Y%m%d_%H%M%S).sql

# Model files backup
tar -czf backups/models_$(date +%Y%m%d).tar.gz models/

# Configuration backup
tar -czf backups/config_$(date +%Y%m%d).tar.gz config/ .env

# Upload to cloud storage
aws s3 sync backups/ s3://your-backup-bucket/news-analysis/
```

## Scaling and Load Balancing

### Horizontal Scaling

#### 1. Kubernetes Deployment
Create `k8s/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: news-analysis
spec:
  replicas: 3
  selector:
    matchLabels:
      app: news-analysis
  template:
    metadata:
      labels:
        app: news-analysis
    spec:
      containers:
      - name: news-analysis
        image: news-analysis:latest
        ports:
        - containerPort: 8501
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: ENVIRONMENT
          value: "production"
        livenessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /_stcore/health
            port: 8501
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: news-analysis-service
spec:
  selector:
    app: news-analysis
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8501
  type: LoadBalancer
```

#### 2. Auto-scaling Configuration
```yaml
# k8s/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: news-analysis-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: news-analysis
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Performance Optimization

#### 1. Caching Layer
```python
# cache.py
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='redis', port=6379, db=0)

def cache_result(expiration=3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"

            # Try to get from cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)

            # Execute function and cache result
            result = func(*args, **kwargs)
            redis_client.setex(cache_key, expiration, json.dumps(result))
            return result
        return wrapper
    return decorator
```

#### 2. Connection Pooling
```python
# database.py
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

---

*This deployment guide provides comprehensive instructions for deploying the News Analysis application across various environments. Choose the deployment strategy that best fits your requirements and infrastructure.*