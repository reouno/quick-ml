# Quick ML APIs

# Build
```
git submodule update --init --recursive
docker build -t quick-ml -f Dockerfile.prod .
```

# Prepare
Prepare AI models according to [models/README.md](models/README.md)

# Run
```
# better
bin/dev up --build

docker rm --rm -p 8001:8001 quick-ml
```

# Usage
Open http://localhost:8001/docs

```
# health check
curl http://localhost:8001/
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

