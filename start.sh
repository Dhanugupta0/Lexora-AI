#!/bin/bash
set -e

# Start FastAPI backend in background
uvicorn app.main:app --host 0.0.0.0 --port 10000 &
API_PID=$!

# Wait for the API to be ready
echo "Waiting for API to start..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:10000/api/v1/health > /dev/null 2>&1; then
        echo "API is ready."
        break
    fi
    sleep 1
done

# Start Gradio frontend (points to local API)
API_URL=http://localhost:10000 python frontend.py &
FRONTEND_PID=$!

# Wait for either process to exit
wait -n $API_PID $FRONTEND_PID
