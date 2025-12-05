# Use an official Python runtime as a parent image
FROM python:3.11-slim-bookworm

# Install Node.js
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# --- Python Agent Setup ---
COPY agent/requirements.txt ./agent/requirements.txt
RUN pip install --no-cache-dir -r agent/requirements.txt

# --- Node.js Frontend Setup ---
COPY package.json package-lock.json ./
RUN npm ci

# Copy the rest of the application code
COPY . .

# Build the Next.js application
RUN npm run build

# Expose the port the app runs on
ENV PORT=3000
EXPOSE 3000

# Make the start script executable
RUN chmod +x start.sh

# Define the command to run the app
CMD ["./start.sh"]
