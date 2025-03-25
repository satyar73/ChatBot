#!/bin/sh

# Copy index.html to fixed-index.html
echo "Creating fixed-index.html from index.html..."
cp /usr/share/nginx/html/index.html /usr/share/nginx/html/fixed-index.html

# Fix paths in fixed-index.html
echo "Fixing paths in fixed-index.html..."
sed -i 's|href="/static|href="./static|g' /usr/share/nginx/html/fixed-index.html
sed -i 's|src="/static|src="./static|g' /usr/share/nginx/html/fixed-index.html
sed -i 's|href="/manifest.json"|href="./manifest.json"|g' /usr/share/nginx/html/fixed-index.html
sed -i 's|href="/logo|href="./logo|g' /usr/share/nginx/html/fixed-index.html

# Fix base href for HashRouter
echo "Adding base href..."
sed -i 's|<head>|<head>\n  <base href="/">|g' /usr/share/nginx/html/fixed-index.html

# Create a copy of the index.html for all routes to handle deep linking
echo "Creating copies for static routing..."
cp /usr/share/nginx/html/fixed-index.html /usr/share/nginx/html/gdrive-indexing.html
cp /usr/share/nginx/html/fixed-index.html /usr/share/nginx/html/testing.html
cp /usr/share/nginx/html/fixed-index.html /usr/share/nginx/html/shopify-indexing.html

# Ensure proper permissions
echo "Setting permissions..."
chmod -R 755 /usr/share/nginx/html

echo "Frontend setup complete!"