FROM python:3.13.2-slim

WORKDIR /app

# Install system dependencies for AWS CLI and nltk
RUN apt-get update && apt-get install -y \
    curl \
    groff \
    less \
    unzip \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install AWS CLI v2
ADD https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip awscliv2.zip
RUN unzip awscliv2.zip \
    && ./aws/install \
    && rm -rf aws awscliv2.zip

COPY brainserve/requirements.txt /app/requirements.txt

COPY models/vectorizer.pkl /app/models/vectorizer.pkl

RUN pip install -r requirements.txt && python -m nltk.downloader stopwords wordnet

COPY brainserve/ /app/

RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--timeout", "120", "app:app"]
