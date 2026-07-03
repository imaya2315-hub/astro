# astro
# Hosting a Fine-tuned Model on a VPS using vLLM

This document explains, step by step, how to host a fine-tuned model
(for example, the Qwen2.5 astrologer model trained with the script in
`scripts/finetune_qwen_astrologer.py`) on a VPS using vLLM.

## Step 1: Choose and Set Up the VPS
- Rent a VPS with a GPU (for example from RunPod, Lambda Labs, Vast.ai,
  or a cloud provider like AWS/GCP if a GPU instance is available). For
  a 7B model, at least 16-24GB GPU VRAM is recommended.
- Choose Ubuntu 22.04 as the OS, since it works well with most ML
  libraries.
- Connect to the VPS using SSH:
  ```bash
  ssh root@your-vps-ip
  ```

## Step 2: Install Basic Requirements
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip git -y
```
Check that the GPU is visible:
```bash
nvidia-smi
```
(Most GPU VPS providers already have NVIDIA drivers and CUDA
pre-installed on the image.)

## Step 3: Create a Virtual Environment
```bash
python3 -m venv vllm-env
source vllm-env/bin/activate
```

## Step 4: Install vLLM
```bash
pip install vllm
```
This also installs the required dependencies (torch, transformers,
etc). Make sure the torch version matches your CUDA version.

## Step 5: Get the Fine-tuned Model
- If the model is saved locally after training, upload it to the VPS:
  ```bash
  scp -r ./qwen_astrologer_merged root@your-vps-ip:/home/models/
  ```
- If it's uploaded to Hugging Face Hub, you can just point vLLM at the
  model name/path directly, no manual copying needed.

## Step 6: Launch the Model with vLLM
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /home/models/qwen_astrologer_merged \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16
```
- `--host 0.0.0.0` makes it reachable from outside, not just localhost.
- `--port 8000` is the port the API runs on (can be changed).
- `--dtype float16` reduces memory usage, useful for limited VRAM.

## Step 7: Open the Port on the VPS Firewall
```bash
sudo ufw allow 8000
```
If using a cloud provider, also open this port in the security group /
firewall settings from the provider's dashboard.

## Step 8: Test the API
```bash
curl http://your-vps-ip:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen_astrologer_merged",
    "prompt": "Hello, how are you?",
    "max_tokens": 50
  }'
```
If it returns a proper response, the model is hosted successfully.

## Step 9: Keep the Server Running
Use a process manager so the server doesn't stop when you close SSH:
```bash
nohup python -m vllm.entrypoints.openai.api_server \
  --model /home/models/qwen_astrologer_merged \
  --host 0.0.0.0 --port 8000 &
```
For production, set this up with `systemd` or `tmux`/`screen` so it
runs in the background and restarts automatically if the VPS reboots.

## Step 10: Add a Reverse Proxy (Optional)
Put Nginx in front of vLLM to handle HTTPS, and connect a domain name
to the VPS IP for a cleaner URL instead of using IP:port directly.

## Step 11: Monitor
```bash
nvidia-smi
tail -f nohup.out
```

That's the full flow — VPS setup, installing vLLM, loading the
fine-tuned model, running the server, and exposing it as an API
endpoint.
