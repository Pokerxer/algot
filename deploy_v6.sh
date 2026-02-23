#!/bin/bash
# V6 Trading Bot - GCP Deployment Script
# Run this on your GCP VM

set -e

ALGOT_DIR="/home/ubuntu/algot"
SERVICE_FILE="/etc/systemd/system/v6-trading.service"

echo "=========================================="
echo "ICT V6 Trading Bot - GCP Deployment"
echo "=========================================="

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "Warning: Running as root is not recommended"
fi

# Clone or update repo
if [ -d "$ALGOT_DIR" ]; then
    echo "[1/5] Updating Algot repository..."
    cd "$ALGOT_DIR"
    git pull
else
    echo "[1/5] Cloning Algot repository..."
    cd ~
    git clone https://github.com/Pokerxer/algot.git "$ALGOT_DIR"
    cd "$ALGOT_DIR"
fi

# Install dependencies
echo "[2/5] Installing dependencies..."
pip3 install -q ib_insync pandas numpy requests python-telegram-bot

# Create log directory
echo "[3/5] Setting up log directory..."
mkdir -p "$ALGOT_DIR/logs"
touch "$ALGOT_DIR/v6_trading.log"

# Copy service file (requires sudo)
echo "[4/5] Installing systemd service..."
if [ -f "$SERVICE_FILE" ]; then
    echo "Service already installed"
else
    sudo cp "$ALGOT_DIR/v6_trading.service" "$SERVICE_FILE"
    sudo systemctl daemon-reload
    sudo systemctl enable v6-trading
fi

# Start the service
echo "[5/5] Starting V6 trading bot..."
sudo systemctl start v6-trading

# Check status
echo ""
echo "=========================================="
echo "V6 Trading Bot Status:"
echo "=========================================="
sudo systemctl status v6-trading --no-pager

echo ""
echo "To view logs:"
echo "  tail -f $ALGOT_DIR/v6_trading.log"
echo ""
echo "To stop:"
echo "  sudo systemctl stop v6-trading"
echo ""
echo "To restart:"
echo "  sudo systemctl restart v6-trading"
