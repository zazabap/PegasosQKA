# Step 1: Use an official Python image as a base
FROM python:3.9-slim

# Step 2: Set the working directory inside the container
WORKDIR ~/Documents/PegasosQKA

# Step 3: Copy the application source code to the container
COPY . ~/Documents/PegasosQKA

# Install dependencies including vim
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    libssl-dev \
    python3-dev \
    build-essential \
    vim  # Added vim installation

# Upgrade pip and install wheel
RUN pip install --upgrade pip && pip install wheel

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


