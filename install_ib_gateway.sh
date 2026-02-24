#!/bin/bash
# IB Gateway Installation and Auto-Start Script for GCP
# Run as: sudo ./install_ib_gateway.sh

set -e

IB_GATEWAY_VERSION="10.25.2a"  # Update to latest version
IB_GATEWAY_DIR="/opt/ibgateway"
IB_GATEWAY_USER="jrwaldehzx"
API_PORT_PAPER=7497
API_PORT_LIVE=7496

echo "=========================================="
echo "IB Gateway Installation for GCP"
echo "=========================================="

# Install Java (required for IB Gateway)
echo "[1/5] Installing Java..."
apt-get update -qq
apt-get install -y -qq openjdk-17-jre-headless wget unzip

# Download IB Gateway
echo "[2/5] Downloading IB Gateway..."
mkdir -p "$IB_GATEWAY_DIR"
cd "$IB_GATEWAY_DIR"

# Note: You'll need to download from IB website manually or use this link
# This is a placeholder - you need to get the actual download link from IB
echo "IMPORTANT: You need to download IB Gateway from Interactive Brokers"
echo "1. Go to https://www.interactivebrokers.com/en/trading/ib gateway.php"
echo "2. Download the Linux version"
echo "3. Upload to your GCP VM"
echo ""
echo "Alternatively, use this workaround to use TWS instead:"
echo "- Install TeamViewer or use GCP serial console for GUI access"
echo "- Or use a VPN to connect to your local TWS"

# Create startup script
echo "[3/5] Creating IB Gateway startup script..."
cat > "$IB_GATEWAY_DIR/start_gateway.sh" << 'SCRIPT'
#!/bin/bash
# IB Gateway startup script for paper trading

export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which java))))
export PATH=$JAVA_HOME/bin:$PATH

cd /opt/ibgateway

# Start IB Gateway in paper trading mode
java -jar -Xmx512m ibgroup/ibs/gw/standalone/IBGateway.jar-paper \
    --port=7497 \
    --api-user=jrwaldehzx \
    --api-password=YOUR_API_PASSWORD \
    --force=true \
    --no guimode &

echo "IB Gateway started on port 7497 (paper)"
SCRIPT

chmod +x "$IB_GATEWAY_DIR/start_gateway.sh"

# Create systemd service
echo "[4/5] Creating systemd service..."
cat > /etc/systemd/system/ib-gateway.service << 'SERVICE'
[Unit]
Description=Interactive Brokers Gateway
After=network.target

[Service]
Type=simple
User=jrwaldehzx
WorkingDirectory=/opt/ibgateway
ExecStart=/opt/ibgateway/start_gateway.sh
Restart=on-failure
RestartSec=30

[Install]
WantedBy=multi-user.target
SERVICE

# Fix permissions
chown -R jrwaldehzx:jrwaldehzx "$IB_GATEWAY_DIR"

echo "[5/5] Installation complete!"
echo ""
echo "NEXT STEPS:"
echo "1. Download IB Gateway from Interactive Brokers"
echo "2. Extract to /opt/ibgateway"
echo "3. Edit /opt/ibgateway/start_gateway.sh with your IB credentials"
echo "4. Run: sudo systemctl enable ib-gateway"
echo "5. Run: sudo systemctl start ib-gateway"
