#!/bin/sh

# Automatically configure AWS CLI using env vars
if [ -n "$AWS_ACCESS_KEY_ID" ] && [ -n "$AWS_SECRET_ACCESS_KEY" ]; then
  mkdir -p ~/.aws

  cat > ~/.aws/credentials <<EOF
[default]
aws_access_key_id = $AWS_ACCESS_KEY_ID
aws_secret_access_key = $AWS_SECRET_ACCESS_KEY
EOF

  cat > ~/.aws/config <<EOF
[default]
region = ${AWS_DEFAULT_REGION:-auto}
output = json
endpoint_url = ${AWS_S3_ENDPOINT_URL}
EOF
fi

# Start the app
exec "$@"
