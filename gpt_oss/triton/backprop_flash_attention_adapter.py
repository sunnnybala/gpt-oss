"""
FlashAttention-style forward/backward Triton kernels with:
- GQA (grouped query attention) mapping
- Sliding Window Attention (SWA)
- Token-based Attention Sinks (prefix tokens always attendable)

Notes and semantics:
- Layout expected by kernels: Q/K/V are [batch, heads, seq_len, head_dim].
- GQA mapping is derived internally by grouping: kv_head_idx = q_head_idx // (H_q / H_kv).
- Sink tokens are defined by an integer `sink_size` (prefix length). This is distinct from
  the repo's existing learned sink scalars used in `gpt_oss.triton.attention`.
- Head dimension must be <= 128 for these kernels.

Convenience adapter provided:
- `flash_swda_with_sink_from_block` accepts tensors shaped like the repo's blocks
  ([batch, seq_len, heads, head_dim]) and calls the kernels with the correct layout.

This module does not register itself in the package exports. Import directly via the file path
or add an export in `gpt_oss/triton/__init__.py` in your own branch if desired.
"""

import torch
import triton
import triton.language as tl
import math
from typing import Optional
@triton.jit
def _flash_attention_forward_swa_kernel(
    # Pointers to Tensors
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr, L_ptr,
    # Stride information for tensors
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    o_stride_b, o_stride_h, o_stride_s,
    m_stride_b, m_stride_h, m_stride_s,
    l_stride_b, l_stride_h, l_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    # Constexpr tile sizes
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Triton kernel for the forward pass of causal FlashAttention with GQA, Sliding Window Attention, and Attention Sink.
    """
    # Identify the block of queries and the batch/head to be processed.
    q_block_idx = tl.program_id(axis=0)
    batch_head_idx = tl.program_id(axis=1)
    
    batch_idx = batch_head_idx // N_Q_HEADS
    q_head_idx = batch_head_idx % N_Q_HEADS

    # --- GQA Logic: Map Query Head to Shared K/V Head ---
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # 2. Initialize accumulators in SRAM.
    m_i = tl.full([BLOCK_M], -float('inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    # 3. Load the block of queries (Q_i).
    q_offsets = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
    q_block = tl.load(q_ptrs, mask=q_offsets[:, None] < SEQ_LEN, other=0.0)
    q_block = tl.cast(q_block, tl.float32)
    qk_scale = softmax_scale * 1.44269504

    window_start = max(0, q_block_idx * BLOCK_M - WINDOW_SIZE)
    # --- Phase 0: Sink blocks that are before the sliding window ---
    sink_end = min(SINK_SIZE, window_start)
    for start_n in range(0, sink_end, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        k_block = tl.cast(k_block, tl.float32)
        s_ij = tl.dot(q_block, k_block, allow_tf32=False)
        s_ij *= qk_scale
        # Attend only to causal sink tokens within sequence length
        valid = (k_offsets[None, :] <= q_offsets[:, None]) & (k_offsets[None, :] < SINK_SIZE) & (k_offsets[None, :] < SEQ_LEN)
        s_ij = tl.where(valid, s_ij, -float('inf'))
        # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        v_block = tl.cast(v_block, tl.float32)


        # 1. Find the new running maximum (`m_new`).
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        # 2. Rescale accumulators; guard all-masked tiles to avoid NaNs.
        no_contrib = m_new == -float('inf')
        alpha = tl.where(no_contrib, 1.0, tl.exp2(m_i - m_new))
        acc_rescaled = acc * alpha[:, None]
        l_i_rescaled = l_i * alpha
        # 3. Compute probabilities safely.
        s_shifted = s_ij - m_new[:, None]
        s_shifted = tl.where(no_contrib[:, None], -float('inf'), s_shifted)
        P_tilde_ij = tl.exp2(s_shifted)
        # 4. Update accumulators.
        l_i = l_i_rescaled + tl.sum(P_tilde_ij, axis=1)
        acc = acc_rescaled + tl.dot(P_tilde_ij, v_block, allow_tf32=False)
        # 5. Update running maximum for next iteration.
        m_i = m_new

    # The kernel should only attend to the `WINDOW_SIZE` most recent key/value tokens.

    # --- Phase 1: Off-Diagonal Blocks (within the window) ---
    for start_n in range(window_start, q_block_idx * BLOCK_M, BLOCK_N):
        #    - A score is invalid if `(query_offset - key_offset) >= WINDOW_SIZE`.
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        k_block = tl.cast(k_block, tl.float32)
        
        # Compute attention scores S_ij = Q_i * K_j^T
        s_ij = tl.dot(q_block, k_block, allow_tf32=False)
        s_ij *= qk_scale
        causal = (k_offsets[None, :] <= q_offsets[:, None])
        within_window = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        in_sink = (k_offsets[None, :] < SINK_SIZE)
        in_range = (k_offsets[None, :] < SEQ_LEN)
        valid = causal & (within_window | in_sink) & in_range
        s_ij = tl.where(valid, s_ij, -float('inf'))
        # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        v_block = tl.cast(v_block, tl.float32)

        # 1. Find the new running maximum (`m_new`).
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        # 2. Rescale accumulators; guard all-masked tiles to avoid NaNs.
        no_contrib = m_new == -float('inf')
        alpha = tl.where(no_contrib, 1.0, tl.exp2(m_i - m_new))
        acc_rescaled = acc * alpha[:, None]
        l_i_rescaled = l_i * alpha
        # 3. Compute probabilities safely.
        s_shifted = s_ij - m_new[:, None]
        s_shifted = tl.where(no_contrib[:, None], -float('inf'), s_shifted)
        P_tilde_ij = tl.exp2(s_shifted)
        # 4. Update accumulators.
        l_i = l_i_rescaled + tl.sum(P_tilde_ij, axis=1)
        acc = acc_rescaled + tl.dot(P_tilde_ij, v_block, allow_tf32=False)
        # 5. Update running maximum for next iteration.
        m_i = m_new

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + \
                 (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_block = tl.load(k_ptrs, mask=k_offsets[None, :] < SEQ_LEN, other=0.0)
        k_block = tl.cast(k_block, tl.float32)
        
        # Compute attention scores S_ij = Q_i * K_j^T
        s_ij = tl.dot(q_block, k_block, allow_tf32=False)
        s_ij *= qk_scale
        # Causal + sliding window + sink mask within the diagonal block
        causal = (k_offsets[None, :] <= q_offsets[:, None])
        within_window = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        in_sink = (k_offsets[None, :] < SINK_SIZE)
        in_range = (k_offsets[None, :] < SEQ_LEN)
        valid = causal & (within_window | in_sink) & in_range
        s_ij = tl.where(valid, s_ij, -float('inf'))
        # Load V_j
        v_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + \
                 (k_offsets[:, None] * v_stride_s + tl.arange(0, HEAD_DIM)[None, :])
        v_block = tl.load(v_ptrs, mask=k_offsets[:, None] < SEQ_LEN, other=0.0)
        v_block = tl.cast(v_block, tl.float32)


        # 1. Find the new running maximum (`m_new`).
        m_ij = tl.max(s_ij, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        # 2. Rescale accumulators; guard all-masked tiles to avoid NaNs.
        no_contrib = m_new == -float('inf')
        alpha = tl.where(no_contrib, 1.0, tl.exp2(m_i - m_new))
        acc_rescaled = acc * alpha[:, None]
        l_i_rescaled = l_i * alpha
        # 3. Compute probabilities safely.
        s_shifted = s_ij - m_new[:, None]
        s_shifted = tl.where(no_contrib[:, None], -float('inf'), s_shifted)
        P_tilde_ij = tl.exp2(s_shifted)
        # 4. Update accumulators.
        l_i = l_i_rescaled + tl.sum(P_tilde_ij, axis=1)
        acc = acc_rescaled + tl.dot(P_tilde_ij, v_block, allow_tf32=False)
        # 5. Update running maximum for next iteration.
        m_i = m_new
       
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    acc = acc / l_i_safe[:, None]
    
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + \
             (q_offsets[:, None] * q_stride_s + tl.arange(0, HEAD_DIM)[None, :])
             
    tl.store(o_ptrs, acc.to(O_ptr.dtype.element_ty), mask=q_offsets[:, None] < SEQ_LEN)
    temp = (q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M))
        # Use (B,H,S) strides when storing per-row stats
    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + temp * m_stride_s
    tl.store(m_ptrs, m_i.to(M_ptr.dtype.element_ty), mask=(temp < SEQ_LEN))
    l_ptrs = L_ptr + batch_idx * l_stride_b + q_head_idx * l_stride_h + temp * l_stride_s
    tl.store(l_ptrs, l_i.to(L_ptr.dtype.element_ty), mask=(temp < SEQ_LEN))

@triton.jit
def _flash_attention_backward_swa_kernel(
    # Inputs
    Q_ptr, K_ptr, V_ptr, O_ptr, M_ptr, L_ptr, dO_ptr,
    # Outputs (grads)
    dQ_ptr, dK_ptr, dV_ptr,
    # Strides for Q/K/V/O (assumed same layout for Q, O, dO, dQ)
    q_stride_b, q_stride_h, q_stride_s,
    k_stride_b, k_stride_h, k_stride_s,
    v_stride_b, v_stride_h, v_stride_s,
    # Strides for M/L (B,H,S)
    m_stride_b, m_stride_h, m_stride_s,
    l_stride_b, l_stride_h, l_stride_s,
    # Kernel parameters
    softmax_scale,
    SEQ_LEN,
    N_Q_HEADS,
    N_KV_HEADS,
    # Constexpr tile sizes
    WINDOW_SIZE: tl.constexpr,
    SINK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    LOG2E = 1.44269504
    qk_scale2 = softmax_scale * LOG2E

    # Program ids
    q_block_idx = tl.program_id(axis=0)
    bh_idx = tl.program_id(axis=1)
    batch_idx = bh_idx // N_Q_HEADS
    q_head_idx = bh_idx % N_Q_HEADS

    # GQA mapping: map q_head to kv_head
    num_groups = N_Q_HEADS // N_KV_HEADS
    kv_head_idx = q_head_idx // num_groups

    # Offsets and pointers for this query block
    q_offsets = q_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    hd = tl.arange(0, HEAD_DIM)
    q_ptrs = Q_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + (q_offsets[:, None] * q_stride_s + hd[None, :])
    o_ptrs = O_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + (q_offsets[:, None] * q_stride_s + hd[None, :])
    do_ptrs = dO_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + (q_offsets[:, None] * q_stride_s + hd[None, :])

    q_mask = q_offsets[:, None] < SEQ_LEN
    q_block = tl.load(q_ptrs, mask=q_mask, other=0.0)
    o_block = tl.load(o_ptrs, mask=q_mask, other=0.0)
    do_block = tl.load(do_ptrs, mask=q_mask, other=0.0)
    q_block = tl.cast(q_block, tl.float32)
    o_block = tl.cast(o_block, tl.float32)
    do_block = tl.cast(do_block, tl.float32)
    # Load stored per-row softmax stats
    m_ptrs = M_ptr + batch_idx * m_stride_b + q_head_idx * m_stride_h + q_offsets * m_stride_s
    l_ptrs = L_ptr + batch_idx * l_stride_b + q_head_idx * l_stride_h + q_offsets * l_stride_s
    m_i = tl.load(m_ptrs, mask=q_offsets < SEQ_LEN, other=0.0)
    l_i = tl.load(l_ptrs, mask=q_offsets < SEQ_LEN, other=1.0)
    l_i = tl.maximum(l_i, 1e-12)

    # delta = sum(dO * O) per row
    delta = tl.sum(do_block * o_block, axis=1)  # (BLOCK_M)

    # Accumulator for dQ
    dQ_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    
    window_start = max(0, q_block_idx * BLOCK_M - WINDOW_SIZE)
    # --- Phase 0: Sink blocks that are before the sliding window ---
    sink_end = min(SINK_SIZE, window_start)
    for start_n in range(0, sink_end, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_offsets < SEQ_LEN 

        # Load K and V once in column-major (D, N), derive row-major via transpose
        k_cols_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        v_cols_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (k_offsets[None, :] * v_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_cols = tl.load(k_cols_ptrs, mask=k_mask[None, :], other=0.0)      # (D, N)
        v_cols = tl.load(v_cols_ptrs, mask=k_mask[None, :], other=0.0)      # (D, N)
        # Cast tiles to fp32 for matmuls
        k_cols = tl.cast(k_cols, tl.float32)
        v_cols = tl.cast(v_cols, tl.float32)
        k_rows = tl.trans(k_cols)  # (N, D)

        # Scores in base-2 domain and probabilities
        q_f32 = tl.cast(q_block, tl.float32)
        s2 = tl.dot(q_f32, k_cols, allow_tf32=False) * qk_scale2  # (M, N)
        causal = (k_offsets[None, :] <= q_offsets[:, None])
        in_sink = (k_offsets[None, :] < SINK_SIZE)
        in_range = (k_offsets[None, :] < SEQ_LEN)
        valid = causal & in_sink & in_range
        s2 = tl.where(valid, s2, -float('inf'))
        p_tilde = tl.exp2(s2 - m_i[:, None])
        P = p_tilde / l_i[:, None]

        # dV partial via matmul: P^T @ dO  => (N,D)
        do_f32 = tl.cast(do_block, tl.float32)
        dv_partial = tl.dot(tl.trans(P), do_f32, allow_tf32=False)

        # t_block = dO @ V^T using V in (D, N)
        t_block = tl.dot(do_f32, v_cols, allow_tf32=False)

        # dS = P * (t_block - delta[:, None])
        dS = P * (t_block - delta[:, None])

        # dQ += dS @ K_rows, scaled by softmax_scale
        dQ_acc += tl.dot(dS, k_rows, allow_tf32=False) * softmax_scale

        # dK partial via matmul: dS^T @ Q, scaled by softmax_scale
        q_f32 = tl.cast(q_block, tl.float32)
        dk_partial = tl.dot(tl.trans(dS), q_f32, allow_tf32=False) * softmax_scale

        # Atomic add into global dV and dK
        dv_ptrs = dV_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (k_offsets[:, None] * v_stride_s + hd[None, :])
        dk_ptrs = dK_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (k_offsets[:, None] * k_stride_s + hd[None, :])
        tl.atomic_add(dv_ptrs, dv_partial, mask=k_mask[:, None])
        tl.atomic_add(dk_ptrs, dk_partial, mask=k_mask[:, None])
    # --- Phase 1: Off-Diagonal Blocks ---
    for start_n in range(window_start, q_block_idx * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_offsets < SEQ_LEN

        # Load K and V once in column-major (D, N), derive row-major via transpose
        k_cols_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        v_cols_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (k_offsets[None, :] * v_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_cols = tl.load(k_cols_ptrs, mask=k_mask[None, :], other=0.0)      # (D, N)
        v_cols = tl.load(v_cols_ptrs, mask=k_mask[None, :], other=0.0)      # (D, N)
        # Cast tiles to fp32 for matmuls
        k_cols = tl.cast(k_cols, tl.float32)
        v_cols = tl.cast(v_cols, tl.float32)
        k_rows = tl.trans(k_cols)  # (N, D)

        # Scores in base-2 domain and probabilities
        q_f32 = tl.cast(q_block, tl.float32)
        s2 = tl.dot(q_f32, k_cols, allow_tf32=False) * qk_scale2  # (M, N)
        causal = (k_offsets[None, :] <= q_offsets[:, None])
        within_window = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        in_sink = (k_offsets[None, :] < SINK_SIZE)
        in_range = (k_offsets[None, :] < SEQ_LEN)
        valid = causal & (within_window | in_sink) & in_range
        s2 = tl.where(valid, s2, -float('inf'))
        p_tilde = tl.exp2(s2 - m_i[:, None])
        P = p_tilde / l_i[:, None]

        # dV partial via matmul: P^T @ dO  => (N,D)
        do_f32 = tl.cast(do_block, tl.float32)
        dv_partial = tl.dot(tl.trans(P), do_f32, allow_tf32=False)

        # t_block = dO @ V^T using V in (D, N)
        t_block = tl.dot(do_f32, v_cols, allow_tf32=False)

        # dS = P * (t_block - delta[:, None])
        dS = P * (t_block - delta[:, None])

        # dQ += dS @ K_rows, scaled by softmax_scale
        dQ_acc += tl.dot(dS, k_rows, allow_tf32=False) * softmax_scale

        # dK partial via matmul: dS^T @ Q, scaled by softmax_scale
        q_f32 = tl.cast(q_block, tl.float32)
        dk_partial = tl.dot(tl.trans(dS), q_f32, allow_tf32=False) * softmax_scale

        # Atomic add into global dV and dK
        dv_ptrs = dV_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (k_offsets[:, None] * v_stride_s + hd[None, :])
        dk_ptrs = dK_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (k_offsets[:, None] * k_stride_s + hd[None, :])
        tl.atomic_add(dv_ptrs, dv_partial, mask=k_mask[:, None])
        tl.atomic_add(dk_ptrs, dk_partial, mask=k_mask[:, None])

    # --- Phase 2: Diagonal Blocks ---
    diag_start = q_block_idx * BLOCK_M
    for start_n in range(diag_start, (q_block_idx + 1) * BLOCK_M, BLOCK_N):
        k_offsets = start_n + tl.arange(0, BLOCK_N)
        k_mask = k_offsets < SEQ_LEN

        k_cols_ptrs = K_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (k_offsets[None, :] * k_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        v_cols_ptrs = V_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (k_offsets[None, :] * v_stride_s + tl.arange(0, HEAD_DIM)[:, None])
        k_cols = tl.load(k_cols_ptrs, mask=k_mask[None, :], other=0.0)
        v_cols = tl.load(v_cols_ptrs, mask=k_mask[None, :], other=0.0)
        k_cols = tl.cast(k_cols, tl.float32)
        v_cols = tl.cast(v_cols, tl.float32)
        k_rows = tl.trans(k_cols)

        # Scores with causal mask inside diagonal tile
        q_f32 = tl.cast(q_block, tl.float32)

        s2 = tl.dot(q_f32, k_cols, allow_tf32=False) * qk_scale2
        causal = (k_offsets[None, :] <= q_offsets[:, None])
        within_window = (q_offsets[:, None] - k_offsets[None, :]) < WINDOW_SIZE
        in_sink = (k_offsets[None, :] < SINK_SIZE)
        in_range = (k_offsets[None, :] < SEQ_LEN)
        valid = causal & (within_window | in_sink) & in_range
        s2 = tl.where(valid, s2, -float('inf'))
        p_tilde = tl.exp2(s2 - m_i[:, None])
        P = p_tilde / l_i[:, None]

        # dV partial P^T @ dO
        do_f32 = tl.cast(do_block, tl.float32)
        dv_partial = tl.dot(tl.trans(P), do_f32, allow_tf32=False)

        # t_block = dO @ V^T
        t_block = tl.dot(do_f32, v_cols, allow_tf32=False)
        t_block = tl.where(valid, t_block, 0.0)

        dS = P * (t_block - delta[:, None])
        dS = tl.where(valid, dS, 0.0)

        dQ_acc += tl.dot(dS, k_rows, allow_tf32=False) * softmax_scale

        q_f32 = tl.cast(q_block, tl.float32)
        dk_partial = tl.dot(tl.trans(dS), q_f32, allow_tf32=False) * softmax_scale

        dv_ptrs = dV_ptr + batch_idx * v_stride_b + kv_head_idx * v_stride_h + (k_offsets[:, None] * v_stride_s + hd[None, :])
        dk_ptrs = dK_ptr + batch_idx * k_stride_b + kv_head_idx * k_stride_h + (k_offsets[:, None] * k_stride_s + hd[None, :])
        tl.atomic_add(dv_ptrs, dv_partial, mask=k_mask[:, None])
        tl.atomic_add(dk_ptrs, dk_partial, mask=k_mask[:, None])

    dQ_ptrs = dQ_ptr + batch_idx * q_stride_b + q_head_idx * q_stride_h + (q_offsets[:, None] * q_stride_s + hd[None, :])
    tl.store(dQ_ptrs, dQ_acc, mask=q_mask)

class FlashSWDAWithSink(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, window_size, sink_size, is_causal=True, softmax_scale=None):
        """
        Forward pass for FlashAttention-style attention with SWA, GQA, and token-based sinks.

        Parameters
        - q: [batch, num_query_heads, seq_len, head_dim]
        - k: [batch, num_kv_heads, seq_len, head_dim]
        - v: [batch, num_kv_heads, seq_len, head_dim]
        - window_size: sliding window length; attends within last `window_size` keys per query
        - sink_size: prefix token count that is always attendable (token-based sinks)
        - is_causal: only causal attention is supported (must be True)
        - softmax_scale: optional scale for QK^T; defaults to 1/sqrt(head_dim)

        Returns
        - o: [batch, num_query_heads, seq_len, head_dim] with same dtype as q
        """
        assert is_causal, "Currently, only causal attention is supported"

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        batch, n_q_heads, seq_len, head_dim = q.shape
        _, n_kv_heads, _, _ = k.shape

        assert q.shape[0] == v.shape[0] and q.shape[2] == v.shape[2] and q.shape[3] == v.shape[3], "Query and Value shapes must be compatible except for num_heads"
        assert k.shape[0] == v.shape[0] and k.shape[1] == v.shape[1] and k.shape[2] == v.shape[2] and k.shape[3] == v.shape[3], "Key and Value shapes must be the same"
        assert head_dim <= 128, "Head dimension must be less than or equal to 128"
        assert n_q_heads % n_kv_heads == 0, "Number of query heads must be divisible by number of K/V heads"

        o = torch.empty_like(q)
        M = torch.empty((batch, n_q_heads, seq_len), device=q.device, dtype=torch.float32)
        L = torch.empty((batch, n_q_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M, BLOCK_N = 128, 64
        grid = (math.ceil(seq_len / BLOCK_M), batch * n_q_heads)

        _flash_attention_forward_swa_kernel[grid](
            q, k, v, o, M, L,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            M.stride(0), M.stride(1), M.stride(2),
            L.stride(0), L.stride(1), L.stride(2),
            softmax_scale,
            seq_len,
            n_q_heads,
            n_kv_heads,
            WINDOW_SIZE=window_size,
            SINK_SIZE=sink_size,
            HEAD_DIM=head_dim,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, o, M, L)
        ctx.softmax_scale = softmax_scale
        ctx.window_size = window_size
        ctx.sink_size = sink_size
        return o

    @staticmethod
    def backward(ctx, do):
        """
        Backward pass computing gradients dQ, dK, dV. Expects
        - do: [batch, num_query_heads, seq_len, head_dim]

        Returns
        - dq, dk, dv with shapes matching q, k, v respectively.
        """
        q, k, v, o, M, L = ctx.saved_tensors
        softmax_scale = ctx.softmax_scale
        window_size = ctx.window_size
        sink_size = ctx.sink_size

        batch, n_q_heads, seq_len, head_dim = q.shape
        n_kv_heads = k.shape[1]

        dq_f = torch.zeros_like(q, dtype=torch.float32)
        dk_rep_f = torch.zeros_like(k, dtype=torch.float32)
        dv_rep_f = torch.zeros_like(v, dtype=torch.float32)

        # Implement backward using a numerically stable streaming algorithm with block-wise keys.
        # This avoids allocating full (seq_len x seq_len) attention tensors.
        softmax_scale = ctx.softmax_scale
        BLOCK_M, BLOCK_N = 128, 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch * n_q_heads)
        _flash_attention_backward_swa_kernel[grid](
        q, k, v, o, M, L, do,
        dq_f, dk_rep_f, dv_rep_f,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        M.stride(0), M.stride(1), M.stride(2),
        L.stride(0), L.stride(1), L.stride(2),
        softmax_scale,
        seq_len,
        n_q_heads,
        n_kv_heads,
        WINDOW_SIZE=window_size,
        SINK_SIZE=sink_size,
        HEAD_DIM=head_dim,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        )

        return dq_f.to(q.dtype), dk_rep_f.to(k.dtype), dv_rep_f.to(v.dtype), None, None, None, None
    
def flash_swda_with_sink(q: torch.Tensor,
                         k: torch.Tensor,
                         v: torch.Tensor,
                         window_size: int,
                         sink_size: int = 0,
                         is_causal: bool = True,
                         scale: Optional[float] = None) -> torch.Tensor:
    """
    Kernel entrypoint expecting layout [B, H, S, D] for Q/K/V.

    See `FlashSWDAWithSink.forward` for a detailed description of parameters.
    """
    return FlashSWDAWithSink.apply(q, k, v, window_size, sink_size, is_causal, scale)


def flash_swda_with_sink_from_block(q_block: torch.Tensor,
                                    k_block: torch.Tensor,
                                    v_block: torch.Tensor,
                                    window_size: int,
                                    sink_size: int = 0,
                                    is_causal: bool = True,
                                    scale: Optional[float] = None) -> torch.Tensor:
    """
    Convenience adapter for tensors shaped like this repo's blocks.

    Inputs
    - q_block: [batch, seq_len, num_query_heads, head_dim]
    - k_block: [batch, seq_len, num_kv_heads, head_dim]
    - v_block: [batch, seq_len, num_kv_heads, head_dim]

    Behavior
    - Permutes to [B, H, S, D], calls the kernels, and permutes back to [B, S, H, D].
    - Enforces head_dim <= 128 and that num_query_heads is divisible by num_kv_heads (GQA).
    - `sink_size` is token-based (prefix length), distinct from learned sink scalars elsewhere in the repo.
    """
    assert q_block.ndim == 4 and k_block.ndim == 4 and v_block.ndim == 4, "Expected 4D tensors"
    bq, sq, hq, d = q_block.shape
    bk, sk, hk, dk = k_block.shape
    bv, sv, hv, dv = v_block.shape
    assert bq == bk == bv and sq == sk == sv, "Batch/sequence mismatch between Q/K/V"
    assert dk == dv == d, "Head dim mismatch between Q/K/V"
    assert hq % hk == 0, "num_query_heads must be divisible by num_kv_heads"
    assert d <= 128, "head_dim must be <= 128 for these kernels"

    q = q_block.permute(0, 2, 1, 3).contiguous()  # [B, Hq, S, D]
    k = k_block.permute(0, 2, 1, 3).contiguous()  # [B, Hk, S, D]
    v = v_block.permute(0, 2, 1, 3).contiguous()  # [B, Hk, S, D]

    o = flash_swda_with_sink(q, k, v, window_size=window_size, sink_size=sink_size, is_causal=is_causal, scale=scale)
    o = o.permute(0, 2, 1, 3).contiguous()  # [B, S, Hq, D]
    return o