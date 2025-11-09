import torch, contextlib

@contextlib.contextmanager
def fp32_strict():
    """
    True FP32 (no TF32). Reproducible baseline on Ampere.
    """
    prev_mm = torch.get_float32_matmul_precision()
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev_mm)
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

@contextlib.contextmanager
def fp32_tf32():
    """
    'FP32' with TF32 fast path enabled on Ampere Tensor Cores.
    """
    prev_mm = torch.get_float32_matmul_precision()
    prev_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.set_float32_matmul_precision("high")   # allow relaxed precision (TF32)
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev_mm)
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32

@contextlib.contextmanager
def amp_fp16():
    """
    AMP autocast to float16.
    """
    assert torch.cuda.is_available(), "CUDA required for AMP-FP16."
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        yield

@contextlib.contextmanager
def amp_bf16():
    """
    AMP autocast to bfloat16 (if supported). Otherwise, graceful no-op.
    """
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        yield
        return
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        yield
