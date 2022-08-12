#!/bin/sh
set -eux

docker build -t quick-ml -f Dockerfile.prod .

