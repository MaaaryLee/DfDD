$ErrorActionPreference = "Stop"

git submodule update --init --recursive

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.11 -m venv .venv
}

& .venv\Scripts\python.exe rtx3080_pipeline\prepare_vendor.py
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& .venv\Scripts\python.exe -m pip install -r requirements-rtx3080.txt

& .venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0)); print('CUDA:', torch.version.cuda)"
& .venv\Scripts\python.exe rtx3080_pipeline\run_pipeline.py @args
