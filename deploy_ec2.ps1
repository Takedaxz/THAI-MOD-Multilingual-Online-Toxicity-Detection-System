$EC2_IP = "54.151.146.86"
$KEY_PATH = "ThaiModKey2.pem"
$USER = "ubuntu"
$REMOTE_TARGET = "$USER@$EC2_IP"

Write-Host "Waiting 10 seconds for SSH server to fully wake up..."
Start-Sleep -Seconds 10

Write-Host "1. Testing SSH Connection & Setting up Server (Updating apt and installing Python 3.11)..."
ssh -i $KEY_PATH -o StrictHostKeyChecking=no $REMOTE_TARGET "
    sudo apt update && sudo apt upgrade -y &&
    sudo apt install -y python3.11-venv python3.11-dev git-lfs &&
    git clone https://github.com/Takedaxz/THAI-MOD-Multilingual-Online-Toxicity-Detection-System.git thai_mod || echo 'Repo already exists' &&
    cd thai_mod &&
    git lfs pull &&
    python3.11 -m venv venv &&
    source venv/bin/activate &&
    pip install -r requirements.txt
"

Write-Host "2. Uploading local .env file..."
scp -i $KEY_PATH -o StrictHostKeyChecking=no .env "$REMOTE_TARGET`:~/thai_mod/.env"

Write-Host "3. Uploading trained weights (LR Baseline) to the server..."
ssh -i $KEY_PATH -o StrictHostKeyChecking=no $REMOTE_TARGET "mkdir -p ~/thai_mod/models"
scp -i $KEY_PATH -o StrictHostKeyChecking=no models/thai_mod_baseline.joblib "$REMOTE_TARGET`:~/thai_mod/models/thai_mod_baseline.joblib"
scp -i $KEY_PATH -o StrictHostKeyChecking=no models/thai_mod_baseline.metadata.json "$REMOTE_TARGET`:~/thai_mod/models/thai_mod_baseline.metadata.json"

Write-Host "4. Starting the Application on the Server (Port 8000)..."
ssh -i $KEY_PATH -o StrictHostKeyChecking=no $REMOTE_TARGET "
    cd thai_mod &&
    source venv/bin/activate &&
    pkill uvicorn || echo 'uvicorn is not running' &&
    nohup uvicorn src.thai_mod_api.main:app --host 0.0.0.0 --port 8000 > server.log 2>&1 &
"

Write-Host "✅ Deployment Complete! The app is running."
Write-Host "Visit: http://$EC2_IP:8000"
