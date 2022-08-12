# Quick ML APIs

# Build
```
docker build -t quick-ml -f Dockerfile.prod .
```

# Run
```
docker rm --rm -p 8000:8000 quick-ml
```

# Usage
Open http://localhost:8000/docs

```
# health check
curl http://localhost:8000/
```

# Development

## Environment
- python3.9

```
# Install packages
pip install -r requirements.txt

# Start server
cd ./app
uvicorn main:app --reload
```

Open http://localhost:8000/docs

