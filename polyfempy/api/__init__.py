# Fix UTF-8 encoding for Windows console (prevents Unicode math symbols from showing as garbled text)
# Fix OpenMP library conflicts (prevents libiomp5md.dll vs libomp.dll conflicts)
import sys
import os
if sys.platform == 'win32':
    try:
        import io
        # Set Python stdout/stderr to UTF-8
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'buffer'):
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        # Set console code page to UTF-8 (Windows-specific)
        os.system('chcp 65001 >nul 2>&1')
    except Exception:
        pass
    
    # Fix OpenMP library conflicts (common when PyTorch and other libraries both link OpenMP)
    # This allows the program to continue, though ideally only one OpenMP runtime should be linked
    if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from .solve import solve
from .config import SimulationConfig
from .result import Result
from .selection import Selection
from .batch import batch_solve

__all__ = ["solve", "SimulationConfig", "Result", "Selection", "batch_solve"]
 