"""
Copyright (c) 2023 by FlashInfer team.
 
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 
  http://www.apache.org/licenses/LICENSE-2.0
 
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
 
import pytest
import torch
from jit_utils import jit_prefill_attention_func_args
from torch.profiler import profile, record_function, ProfilerActivity
import flashinfer
import math
 
import aiter
from einops import rearrange, repeat
 
 
@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if flashinfer.jit.has_prebuilt_ops:
        yield
    else:
        try:
            flashinfer.jit.parallel_load_modules(
                jit_prefill_attention_func_args(
                    [torch.float16],  # q_dtypes
                    [torch.float16],  # kv_dtypes
                    [128, 256],  # head_dims
                    [0, 1, 2],  # pos_encoding_modes
                    [False],  # use_sliding_windows
                    [False, True],  # use_logits_soft_caps
                    [False],  # allow_fp16_qk_reductions
                )
            )
        except Exception as e:
            # abort the test session if warmup fails
            pytest.exit(str(e))
        finally:
            yield
 
def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
    key_leftpad=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    if key_leftpad is not None:
        key_leftpad = rearrange(key_leftpad, "b -> b 1 1 1")
        col_idx = repeat(col_idx, "s -> b 1 1 s", b=key_leftpad.shape[0])
        col_idx = torch.where(col_idx >= key_leftpad, col_idx - key_leftpad, 2**32)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )
 
def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    causal: bool = False,
    window_left: int = -1,
    logits_soft_cap: float = 0.0
) -> torch.Tensor:
    if causal:
        window_size = (window_left, 0)
    else:
        window_size = (-1, -1)
 
    head_dim = query.shape[2]
    seqlen_q = query.shape[0]
    seqlen_k = key.shape[0]
    scale = 1.0 / math.sqrt(head_dim)
 
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query.float(), key.float())
    if 0 < logits_soft_cap:
        attn_weights = logits_soft_cap * torch.tanh(attn_weights / logits_soft_cap)
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            device=query.device,
        )
        attn_weights.masked_fill_(local_mask, float("-inf"))
    attn_weights = torch.softmax(attn_weights, dim=-1)
    if window_size[0] >= 0 or window_size[1] >= 0:
        attn_weights = attn_weights.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value.float())
    return out.to(query)
 
 
 
@pytest.mark.parametrize("batch_size", [1, 7])
@pytest.mark.parametrize("qo_len,kv_len", [
    (8192, 8192),
    (16384, 16384),
    (32768, 32768),
    (4095, 8193),
    (1, 8193),
])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("num_qo_heads,num_kv_heads", [(6, 1), (3, 1)])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("kv_layout", ["NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE"])
@pytest.mark.parametrize("use_cuda_graph", [False])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
@pytest.mark.parametrize("return_lse", [False])
@pytest.mark.parametrize("contiguous_kv", [True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("q_init_min,q_init_max", [(-10, 10)])
@pytest.mark.parametrize("kv_init_min,kv_init_max", [(-10, 10)])
@pytest.mark.parametrize("seed", [123])
def test_batch_prefill_with_paged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    causal,
    kv_layout,
    pos_encoding_mode,
    use_cuda_graph,
    logits_soft_cap,
    return_lse,
    contiguous_kv,
    dtype,
    q_init_min,
    q_init_max,
    kv_init_min,
    kv_init_max,
    seed
):
    if seed is not None:
        torch.manual_seed(seed)
 
    if causal and kv_len < qo_len:
        pytest.skip('kv_len < qo_len is not allowed if causal=True')
 
    if head_dim == 64 and qo_len <= 64:
       pytest.skip('Unsupported configuration')
 
    def create_tensor(min, max, *args, **kwargs):
        x = torch.randn(*args, **kwargs)
        x = (x - x.min()) / (x.max() - x.min())
        return (min + (max - min) * x)
 
    def convert_lens_to_indtpr(lens):
        return torch.cumsum(torch.cat((torch.tensor([0]), lens)), dim=0).int()
 
    q = create_tensor(q_init_min, q_init_max, batch_size * qo_len, num_qo_heads, head_dim, dtype=dtype).to(0)
    if 1 < batch_size:
        qo_lens = torch.randint(1, qo_len + 1, (batch_size,)).int()
    else:
        qo_lens = torch.full((batch_size,), qo_len).int()
    q_indptr_cpu = convert_lens_to_indtpr(qo_lens)
    max_num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = max_num_pages_per_seq * batch_size
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = create_tensor(kv_init_min, kv_init_max, *kv_shape, dtype=torch.float32).to(0)
        kv_data = kv_data_fp32.to(dtype)
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = create_tensor(kv_init_min, kv_init_max, *kv_shape, dtype=torch.float32).to(0)
        kv_data = kv_data_fp32.to(dtype)
    if 1 < batch_size:
        kv_lens = torch.maximum(qo_lens,
            torch.randint(1, kv_len + 1, (batch_size,))).int()
    else:
        kv_lens = torch.full((batch_size,), kv_len).int()
    kv_num_used_pages = (kv_lens + page_size - 1) // page_size
    kv_indptr_cpu = convert_lens_to_indtpr(kv_num_used_pages)
    kv_indices_cpu = torch.randperm(total_num_pages).int()
    kv_last_page_len_cpu = ((kv_lens  - 1) % page_size + 1).int()
 
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    if not use_cuda_graph:
        q_indptr_gpu = q_indptr_cpu.to(0)
        kv_indptr_gpu = kv_indptr_cpu.to(0)
        kv_indices_gpu = kv_indices_cpu.to(0)
        kv_last_page_len_gpu = kv_last_page_len_cpu.to(0)
        wrapper = flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper(
            workspace_buffer, kv_layout
        )
        wrapper.plan(
            q_indptr_gpu,
            kv_indptr_gpu,
            kv_indices_gpu,
            kv_last_page_len_gpu,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=causal,
            pos_encoding_mode=pos_encoding_mode,
            logits_soft_cap=logits_soft_cap,
            q_data_type=dtype
        )
        for _ in range(10):
            if return_lse:
                o, _ = wrapper.run(q, kv_data, return_lse=True)
            else:
                o = wrapper.run(q, kv_data)
           
        # k_cache =    kv_data[:,0,:,:,:],#k_buffer
        # v_cache =    kv_data[:,1,:,:,:],#v_buffer
       
        chunks = torch.chunk(kv_data, 2, dim=1)
        k_cache = chunks[0].squeeze(2).squeeze(2)  
        v_cache = chunks[1].squeeze(2).squeeze(2)
        # def flash_attn_param_gen_from_extend(kv_indptr, kv_indices, k_buffer, v_buffer):
        #     # B, H_KV, D, dtype
        #     H_KV = k_buffer.shape[1]
        #     D = k_buffer.shape[2]
        #     dtype = k_buffer.dtype
        #     i = 0
 
        #     k_unpad = torch.empty((kv_indptr[-1], H_KV, D), dtype=dtype, device="cuda")
        #     v_unpad = torch.empty((kv_indptr[-1], H_KV, D), dtype=dtype, device="cuda")
        #     k_unpad[kv_indptr[i] : kv_indptr[i + 1]] = k_buffer[kv_indices[kv_indptr[i]:kv_indptr[i + 1]]]
        #     v_unpad[kv_indptr[i] : kv_indptr[i + 1]] = v_buffer[kv_indices[kv_indptr[i]:kv_indptr[i + 1]]]
        #     return k_unpad, v_unpad
        # k, v = flash_attn_param_gen_from_extend(kv_indptr_gpu, kv_indices_gpu, k_cache, v_cache)
        o_ck_flash_attn = aiter.flash_attn_varlen_func(
            q,
            k_cache,
            v_cache,
            q_indptr_gpu, #qo_indptr,
            kv_indptr_gpu, #qo_indptr + kv_indptr,
            8192, #max_len_extend,
            8192, #max_len_in_batch,
            causal=True,
            alibi_slopes=None,
            return_lse=False,
            return_attn_probs=False,
            block_table=kv_indices_gpu
        )[0]
        print(o_ck_flash_attn.shape)
        print(o.shape)
        # o = o_ck_flash_attn
        print("maxo:",torch.max(torch.abs(o)))
        print("maxdiff:",torch.max(torch.abs(o_ck_flash_attn - o)))
 
 
if __name__ == "__main__":
    default_batch_prefill_with_paged_kv_cache_params = dict(
        batch_size=1,
        kv_len=8192,
        qo_len=8192,
        page_size=1,
        num_qo_heads=6,
        num_kv_heads=1,
        head_dim=128,
        causal=True,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        use_cuda_graph=False,
        logits_soft_cap=0.0,
        return_lse=False,
        contiguous_kv=True,
        dtype=torch.float16,
        q_init_min=-3,
        q_init_max=3,
        kv_init_min=-3,
        kv_init_max=3,
        seed=19378,
    )
 
    softcap_batch_prefill_with_paged_kv_cache_params = \
        default_batch_prefill_with_paged_kv_cache_params | dict(logits_soft_cap=30.0)
 
    test_batch_prefill_with_paged_kv_cache(
        **softcap_batch_prefill_with_paged_kv_cache_params
    )
 
