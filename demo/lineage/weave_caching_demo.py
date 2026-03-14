#!/usr/bin/env python
"""Demo: Conditioning cached when pretrain changes.

Weave tracks calls but doesn't auto-skip. We use weave.ref() to check cache.

Usage:
    python demo/lineage/weave_caching_demo.py
    # Edit pretrain_step, run again → conditioning from cache
"""

import weave
from pydantic import BaseModel


class ConditioningOutput(BaseModel):
    cell_types: list[str]
    vocab_size: int


class PretrainOutput(BaseModel):
    checkpoint_path: str
    step: int


weave.init("hooke-caching-demo")


# =============================================================================
# Cached step wrapper
# =============================================================================


def cached_op(name: str):
    """Decorator: check cache before running, publish result after."""
    def decorator(fn):
        @weave.op()
        def wrapper(*args, **kwargs):
            # Try to get cached result
            try:
                cached = weave.ref(f"{name}:latest").get()
                print(f"⏭️  {name}: CACHED")
                return cached
            except Exception:
                pass  # Not cached
            
            # Run and cache
            print(f"🔧 {name}: RUNNING...")
            result = fn(*args, **kwargs)
            weave.publish(result, name=name)
            return result
        
        wrapper.__name__ = fn.__name__
        return wrapper
    return decorator


# =============================================================================
# Steps
# =============================================================================


@cached_op("conditioning")
def conditioning_step() -> ConditioningOutput:
    """Hash independent of pretrain_step."""
    return ConditioningOutput(
        cell_types=["ARPE19", "HUVEC"],
        vocab_size=2048,
    )


@cached_op("pretrain")
def pretrain_step(config: ConditioningOutput) -> PretrainOutput:
    """Change this → cache miss. But conditioning still cached."""
    
    # =============================================
    # MODIFY THIS TO BUST PRETRAIN CACHE:
    step = 300000  # ← Change this value
    # =============================================
    
    return PretrainOutput(
        checkpoint_path="/checkpoints/model.pt",
        step=step,
    )


# =============================================================================
# Pipeline
# =============================================================================


def main():
    print("\n" + "=" * 60)
    print("Pipeline Run")
    print("=" * 60 + "\n")
    
    config = conditioning_step()
    print(f"  → {config}\n")
    
    result = pretrain_step(config)
    print(f"  → {result}\n")
    
    print("=" * 60)
    print("""
To test:
1. Run once → both RUNNING
2. Run again → both CACHED  
3. Delete pretrain cache, run → conditioning CACHED, pretrain RUNNING

Delete cache:
    # Via W&B UI or:
    weave.ref("pretrain:latest").delete()
    """)


if __name__ == "__main__":
    main()
