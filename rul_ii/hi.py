# hi.py
import numpy as np
import pandas as pd
from cfar import cfar_hits_cellwise, count_events_from_hits

def build_hi_for_channel_cellwise(stream_iter,
                                  fs_hz:int,
                                  cell_size:int,
                                  pfa:float,
                                  n_guard_cells:int,
                                  n_train_cells:int):
    """
    Processes a channel as a stream of chunks, running CFAR on **cell power**.
    Robust to chunk boundaries via carry (partial cell) + prev_on (event continues).
    Returns DataFrame ['t_sec','cum_hits'] sampled at end of each processed chunk.
    """
    cum = 0
    t_list, h_list = [], []
    carry = None        # leftover samples < cell_size from previous chunk
    prev_on = False     # last cell of previous chunk was inside an event?

    for base_idx, seg in stream_iter:
        if carry is not None and carry.size:
            seg = np.concatenate([carry, seg])
        hits_cell, used, _ = cfar_hits_cellwise(
            seg, pfa=pfa, cell_size=cell_size,
            n_guard_cells=n_guard_cells, n_train_cells=n_train_cells
        )
        # count events in this chunk (avoid double counting across chunks)
        ev, prev_on = count_events_from_hits(hits_cell, prev_on)
        cum += ev

        # keep leftover for next chunk
        carry = seg[used:] if used < len(seg) else seg[:0]

        # timeline at the end of used samples
        t_list.append((base_idx + used) / fs_hz)
        h_list.append(cum)

    return pd.DataFrame({"t_sec": t_list, "cum_hits": h_list})
