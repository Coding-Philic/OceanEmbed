#!/usr/bin/env bash
# ==============================================================================
# 🌊 OceanEmbed — Production VPS Deployment Script (Ubuntu / AWS t3.small)
# ==============================================================================
set -e

echo "=================================================================="
echo "🚀 Starting OceanEmbed Production VPS Setup (AWS t3.small)"
echo "=================================================================="

# 1. Setup 4 GB Swap (Essential for t3.small 2 GB RAM to prevent Out-Of-Memory)
SWAP_TOTAL=$(free -m | awk '/Swap:/ {print $2}')
if [ "$SWAP_TOTAL" -lt 2000 ]; then
    echo "⚙️ Creating 4 GB swap space for stable 2 GB RAM execution..."
    sudo fallocate -l 4G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=4096
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    if ! grep -q '/swapfile' /etc/fstab; then
        echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    fi
    echo "✅ 4 GB Swap successfully created!"
else
    echo "✅ Sufficient swap already active ($SWAP_TOTAL MB)."
fi

# 2. Update System & Install Core Packages
echo "📦 Installing system dependencies (Nginx, Python3, Git)..."
sudo apt-get update -y
sudo apt-get install -y nginx git python3-pip python3-venv curl ufw

# 3. Setup Python Virtual Environment
APP_DIR="/home/ubuntu/OceanEmbed"
cd "$APP_DIR"

if [ ! -d "venv" ]; then
    echo "🐍 Creating Python virtual environment in $APP_DIR/venv..."
    python3 -m venv venv
fi

echo "📦 Installing lightweight Python dependencies..."
"$APP_DIR/venv/bin/pip" install --upgrade pip setuptools wheel
"$APP_DIR/venv/bin/pip" install -r "$APP_DIR/webapp/requirements.txt"

# Install CPU-only PyTorch (only ~150 MB instead of 3.5 GB CUDA bundle)
echo "⚡ Installing CPU-optimized PyTorch..."
"$APP_DIR/venv/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Ensure Outputs Directory Exists
mkdir -p "$APP_DIR/outputs"

# 5. Configure Nginx Reverse Proxy
echo "🌐 Configuring Nginx reverse proxy (Port 80)..."
sudo cp "$APP_DIR/deploy/nginx_oceanembed.conf" /etc/nginx/sites-available/oceanembed
sudo rm -f /etc/nginx/sites-enabled/default
sudo ln -sf /etc/nginx/sites-available/oceanembed /etc/nginx/sites-enabled/oceanembed

sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx
echo "✅ Nginx is running and listening on Port 80!"

# 6. Configure & Start Systemd Service
echo "⚙️ Setting up OceanEmbed systemd service..."
sudo cp "$APP_DIR/deploy/oceanembed.service" /etc/systemd/system/oceanembed.service
sudo systemctl daemon-reload
sudo systemctl enable oceanembed
sudo systemctl restart oceanembed

# 7. Print Status and Public IP
PUBLIC_IP=$(curl -s http://checkip.amazonaws.com || curl -s ifconfig.me || echo "YOUR_AWS_PUBLIC_IP")

echo "=================================================================="
echo "🎉 DEPLOYMENT COMPLETE!"
echo "=================================================================="
echo "🌍 Website URL: http://$PUBLIC_IP/"
echo "📊 Service Status:"
sudo systemctl status oceanembed --no-pager -n 5
echo "=================================================================="
echo "To view live web logs: sudo journalctl -u oceanembed -f"
echo "=================================================================="
