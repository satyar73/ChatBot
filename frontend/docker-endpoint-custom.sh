#!/bin/sh

# Define backend URL with environment variable support
# The backend service URL can be configured via environment variable
BACKEND_SERVICE_HOST=${BACKEND_SERVICE_HOST:-"backend"}
BACKEND_SERVICE_PORT=${BACKEND_SERVICE_PORT:-"8005"}
CONTAINER_BACKEND="${BACKEND_SERVICE_HOST}:${BACKEND_SERVICE_PORT}"

# Function to check if a backend is available
check_backend() {
  echo "Checking backend at $1..."
  nc -z -w1 $(echo $1 | cut -d: -f1) $(echo $1 | cut -d: -f2) 2>/dev/null
  return $?
}

# For AWS ECS, we rely primarily on service discovery
# If the service isn't available, we'll just use the defined service name
# as ECS will eventually make it available
if check_backend $CONTAINER_BACKEND; then
  echo "Using backend at $CONTAINER_BACKEND"
  BACKEND_URL=$CONTAINER_BACKEND
else
  echo "Backend not immediately available, will use service name: $CONTAINER_BACKEND"
  # In AWS ECS, the service should become available, so we'll use the same name
  BACKEND_URL=$CONTAINER_BACKEND
fi

# Generate Nginx config
cat > /etc/nginx/conf.d/default.conf << EOF
server {
    listen 80;
    server_name localhost;

    # Global proxy settings
    proxy_read_timeout 300;
    proxy_connect_timeout 300;
    proxy_send_timeout 300;

    # Add DNS resolver for host.docker.internal
    resolver 127.0.0.11 valid=30s ipv6=off;

    # Set backend variable
    set \$backend_server "${BACKEND_URL}";

    # Serve static React files
    location / {
        root /usr/share/nginx/html;
        index index.html fixed-index.html;
        try_files \$uri \$uri/ /index.html /fixed-index.html =404;

        # Add headers for static files
        add_header X-Served-By "Frontend Nginx" always;
        add_header Cache-Control "no-cache, no-store, must-revalidate" always;

        # Add debug headers
        add_header X-Debug-Path \$uri always;
        add_header X-Backend-Server \$backend_server always;
    }

    # Handle direct route access
    location ~ ^/(gdrive-indexing|testing|shopify-indexing|create-slides)$ {
        root /usr/share/nginx/html;
        try_files /\$1.html /fixed-index.html /index.html =404;
        add_header X-Served-By "Frontend Nginx - Routes" always;
        add_header Cache-Control "no-cache, no-store, must-revalidate" always;
    }

    # Explicitly handle static files
    location /static/ {
        alias /usr/share/nginx/html/static/;
        add_header Cache-Control "public, max-age=31536000" always;
    }

    # Chat API
    location /chat {
        # Handle OPTIONS preflight requests
        if (\$request_method = 'OPTIONS') {
            add_header 'Access-Control-Allow-Origin' '*' always;
            add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS, PUT, DELETE' always;
            add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization' always;
            add_header 'Access-Control-Max-Age' 1728000;
            add_header 'Content-Type' 'text/plain; charset=utf-8';
            add_header 'Content-Length' 0;
            return 204;
        }

        # Proxy to dynamic backend
        proxy_pass http://\$backend_server/chat;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # CORS headers for responses
        add_header 'Access-Control-Allow-Origin' '*' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, OPTIONS, PUT, DELETE' always;
        add_header 'Access-Control-Allow-Headers' 'Origin, X-Requested-With, Content-Type, Accept, Authorization' always;
    }

    # Add similar configurations for other API endpoints
    # ...rest of your nginx.conf
}
EOF

# Run the standard Nginx entrypoint
exec /docker-entrypoint.sh nginx -g "daemon off;"