FROM python:3.11-slim

# Metadata
LABEL maintainer="HIV RL Research"
LABEL description="HIV Drug Sequencing RL Environment — OpenEnv Challenge"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy requirements first (layer caching)
COPY --chown=user requirements.txt $HOME/app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY --chown=user . $HOME/app/

# Environment variable defaults (can be overridden)
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4.1-mini"
ENV PORT=7860

# Expose port for Hugging Face Spaces
EXPOSE 7860

# Default: run the web interface
CMD ["python", "server/app.py"]
