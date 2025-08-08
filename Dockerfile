# Use NVIDIA base image with CUDA 11.8 support
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

LABEL authors="krishna"

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    build-essential \
    curl \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set python3 as default python
RUN ln -sf python3 /usr/bin/python && ln -sf pip3 /usr/bin/pip

# Install PyTorch with CUDA 11.8
RUN pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 torchaudio==2.7.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Clone and install IndicTransToolkit
#RUN git clone https://github.com/VarunGumma/IndicTransToolkit.git  \
#    && cd IndicTransToolkit \
#    && pip install --editable ./ \
#    && cd ..

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

EXPOSE 8000

# Start the FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]