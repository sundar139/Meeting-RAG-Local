from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch
    except Exception as exc:
        print(f"torch_import_error={exc}")
        return 1

    print(f"torch_version={torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"cuda_available={cuda_available}")

    if not cuda_available:
        print("device_name=unavailable")
        return 1

    device_name = torch.cuda.get_device_name(0)
    print(f"device_name={device_name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
