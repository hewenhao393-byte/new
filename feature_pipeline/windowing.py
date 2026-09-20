from __future__ import annotations
from dataclasses import dataclass
from .config import CONFIG

@dataclass(frozen=True)
class Window:
    split:str; start:int; end:int; train_block_id:str|None

def _slide(start,end,split,block=None):
    for pos in range(start,end-CONFIG.window_size+1,CONFIG.step_size):
        yield Window(split,pos,pos+CONFIG.window_size,block)

def iter_temporal_windows(n_samples,bounds):
    ts,te=bounds["train"]
    block=0
    for start in range(ts,te,CONFIG.train_block_samples):
        end=min(start+CONFIG.train_block_samples,te)
        yield from _slide(start,end,"train",f"block_{block:02d}"); block+=1
    yield from _slide(*bounds["test"],"test")
