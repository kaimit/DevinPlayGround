"""
Attention layer implementation in Triton for transformer inference.
"""
import triton
import triton.language as tl

@triton.jit
def attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    batch_size, seq_len, num_heads, head_dim,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)

    # Compute attention scores
    # Load query, key, value blocks
    q = tl.load(q_ptr + pid * BLOCK_SIZE)
    k = tl.load(k_ptr + pid * BLOCK_SIZE)
    v = tl.load(v_ptr + pid * BLOCK_SIZE)

    # Compute scaled dot product attention
    qk = tl.dot(q, k) / tl.sqrt(head_dim)

    # Apply softmax
    qk_max = tl.max(qk, 1)
    exp_qk = tl.exp(qk - qk_max)
    sum_exp = tl.sum(exp_qk, 1)
    softmax = exp_qk / sum_exp

    # Compute attention output
    out = tl.dot(softmax, v)

    # Store result
    tl.store(out_ptr + pid * BLOCK_SIZE, out)

@triton.jit
def mlp_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    hidden_dim, mlp_dim,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)

    # Load input and weights
    x = tl.load(input_ptr + pid * BLOCK_SIZE)
    w = tl.load(weight_ptr + pid * BLOCK_SIZE)
    b = tl.load(bias_ptr + pid * BLOCK_SIZE)

    # Compute MLP forward pass
    h = tl.dot(x, w) + b

    # Apply GELU activation
    gelu = h * 0.5 * (1.0 + tl.tanh(0.797885 * h + 0.035677 * h * h * h))

    # Store result
    tl.store(output_ptr + pid * BLOCK_SIZE, gelu)
