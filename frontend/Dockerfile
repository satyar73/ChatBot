FROM node:20-alpine as build

WORKDIR /app

# Copy package.json and package-lock.json
COPY package.json package-lock.json ./

# Install dependencies
RUN npm ci

# Copy the rest of the frontend code
COPY . .

# Override the proxy in package.json to work in Docker
# This is empty because Nginx will handle the proxying
ENV REACT_APP_API_URL=""

# Add PUBLIC_URL to make sure assets are loaded correctly
ENV PUBLIC_URL="."

# Build the app for production
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy the build output from the build stage
COPY --from=build /app/build /usr/share/nginx/html

# Create setup script
COPY setup-nginx.sh /docker-entrypoint.d/
RUN chmod +x /docker-entrypoint.d/setup-nginx.sh

# Copy custom nginx config to properly handle React Router
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]