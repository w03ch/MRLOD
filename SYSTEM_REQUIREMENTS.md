## Reference software

| Component | Version |
| --- | --- |
| Operating system | Windows 11 Enterprise, 64-bit, build 10.0.26200 |
| Python | 3.9.23 |
| PyTorch | 2.8.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| cuDNN | 9.10.2 |
| NumPy | 1.26.4 |
| pandas | 2.3.1 |
| scikit-learn | 1.6.1 |

## Reference hardware and driver

| Component | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5080, 16 GB |
| NVIDIA driver | 591.86 |
| CPU | AMD Ryzen 7 9800X3D, 8 cores / 16 threads |
| System memory | 48 GB RAM |

## Conda installation

Create environment:

```powershell
conda env create -f environment.yml
```

## Verification

Confirm the runtime before training:

```powershell
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
nvidia-smi
```

Expected output for the reference environment:

```text
PyTorch: 2.8.0+cu128
CUDA runtime: 12.8
GPU: NVIDIA GeForce RTX 5080
NVIDIA driver: 591.86
```
