# HFT Trading Platform - Deployment Guide

This directory contains all deployment configurations for the HFT Trading Platform.

## Directory Structure

```
deploy/
├── docker/                    # Docker configurations
│   ├── Dockerfile.api         # API service
│   ├── Dockerfile.worker      # Worker service
│   ├── Dockerfile.frontend    # Next.js frontend
│   ├── docker-compose.yml     # Development setup
│   ├── docker-compose.prod.yml # Production overrides
│   └── init-db.sql            # Database initialization
├── kubernetes/                # Kubernetes manifests
│   ├── namespace.yaml         # Namespace & quotas
│   ├── configmap.yaml         # Application config
│   ├── secrets.yaml           # Secrets (template)
│   ├── api-deployment.yaml    # API deployment & HPA
│   ├── worker-deployment.yaml # Worker deployments
│   ├── ingress.yaml           # Ingress & TLS
│   └── storage.yaml           # PVCs & Storage classes
└── monitoring/                # Monitoring stack
    ├── prometheus.yml         # Prometheus config
    ├── alerting-rules.yml     # Alert definitions
    └── grafana/               # Grafana dashboards
```

## Quick Start

### Local Development (Docker Compose)

```bash
# Start all services
cd deploy/docker
docker-compose up -d

# View logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Production Deployment (Kubernetes)

```bash
# Create namespace
kubectl apply -f kubernetes/namespace.yaml

# Create secrets (edit first!)
kubectl apply -f kubernetes/secrets.yaml

# Deploy configuration
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/storage.yaml

# Deploy services
kubectl apply -f kubernetes/api-deployment.yaml
kubectl apply -f kubernetes/worker-deployment.yaml

# Setup ingress
kubectl apply -f kubernetes/ingress.yaml
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| API | 8000 | FastAPI trading backend |
| Frontend | 3000 | Next.js dashboard |
| PostgreSQL | 5432 | Main database |
| TimescaleDB | 5433 | Time-series data |
| Redis | 6379 | Caching & pub/sub |
| RabbitMQ | 5672 | Message queue |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3001 | Dashboards |

## Environment Variables

### Required
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `JWT_SECRET_KEY` - Secret for JWT tokens

### Optional
- `ALPACA_API_KEY` - Alpaca API key
- `ALPACA_SECRET_KEY` - Alpaca secret
- `TRADING_MODE` - `paper` or `live`

## Monitoring

### Prometheus Metrics

The API exposes metrics at `/metrics`:
- `hft_portfolio_value` - Current portfolio value
- `hft_portfolio_drawdown` - Current drawdown
- `hft_trades_total` - Total trades executed
- `hft_agent_sharpe_ratio` - Agent performance

### Alerts

Configured alerts include:
- High API latency (>500ms p95)
- API error rate (>5%)
- High drawdown (>10%)
- Circuit breaker triggered
- Database connectivity issues

## Scaling

### Horizontal Pod Autoscaler

The API auto-scales based on:
- CPU utilization (target: 70%)
- Memory utilization (target: 80%)

```yaml
minReplicas: 3
maxReplicas: 10
```

### Manual Scaling

```bash
kubectl scale deployment hft-api --replicas=5 -n hft-trading
```

## Backup & Recovery

### Database Backup

```bash
# Backup
kubectl exec -n hft-trading postgres-0 -- pg_dump -U postgres hft > backup.sql

# Restore
kubectl exec -i -n hft-trading postgres-0 -- psql -U postgres hft < backup.sql
```

### Model Backup

Models are stored in the `model-pvc` volume. Backup regularly:

```bash
kubectl cp hft-trading/hft-api-xxx:/app/models ./models-backup
```

## Security

- All secrets should be managed with external secrets manager (Vault, AWS Secrets Manager)
- TLS termination at ingress level
- Network policies restrict pod-to-pod communication
- RBAC enforced for all Kubernetes resources

## Troubleshooting

### Common Issues

1. **API not starting**
   - Check database connectivity
   - Verify secrets are properly set

2. **High latency**
   - Check Redis connectivity
   - Review database query performance

3. **Model loading fails**
   - Verify model volume is mounted
   - Check model file permissions

### Useful Commands

```bash
# Check pod status
kubectl get pods -n hft-trading

# View logs
kubectl logs -f deployment/hft-api -n hft-trading

# Describe pod
kubectl describe pod <pod-name> -n hft-trading

# Port forward for debugging
kubectl port-forward svc/hft-api-service 8000:80 -n hft-trading
```

