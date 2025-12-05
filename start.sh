#!/bin/bash

# Start the Python agent in the background
echo "Starting Python Agent..."
cd agent
# Use uvicorn directly or the agent script if it uses uvicorn internally
# Based on agent.py, it uses uvicorn.run if __main__, but for production we can run it with uvicorn CLI
# We need to make sure it listens on a different port than the frontend (e.g., 8000)
# The frontend connects to BACKEND_URL, which we should set to http://localhost:8000
uvicorn agent:app --host 0.0.0.0 --port 8000 &

# Wait a bit for the agent to start
sleep 5

# Start the Next.js frontend
echo "Starting Next.js Frontend..."
cd ..
# Next.js start script uses $PORT
npm start
