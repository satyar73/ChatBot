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

# Ensure proper permissions
echo "Setting permissions..."
chmod -R 755 /usr/share/nginx/html

echo "Frontend setup complete!"