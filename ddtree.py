import heapq
import time
from functools import lru_cache
from types import SimpleNamespace

from loguru import logger
import numpy as np
import torch
from transformers import AutoModelForCausalLM, DynamicCache

from model import DFlashDraftModel, sample, extract_context_feature
from dflash import dflash_generate, cuda_time, empty_stage_times


DDTREE_STAGE_ORDER = ("draft", "tree_build", "tree_compile", "verify", "commit")
DDTREE_TREE_BUILD_STAGE_ORDER = ("tree_build_copy", "tree_build_heap", "tree_build_visibility")


_CPP_COMPACT_ENABLED = False


@lru_cache(maxsize=1)
def load_cpp_compact_module():
    try:
        from torch.utils.cpp_extension import load_inline
    except Exception as exc:
        logger.warning(f"torch.utils.cpp_extension is unavailable; falling back to Python cache compaction. {exc}")
        return None

    cpp_source = r"""
torch::Tensor compact_tail_inplace(torch::Tensor cache_tensor, int64_t past_length, torch::Tensor keep_current_indices) {
    TORCH_CHECK(cache_tensor.dim() >= 2, "cache_tensor must have rank >= 2");
    TORCH_CHECK(keep_current_indices.dim() == 1, "keep_current_indices must be a 1D tensor");
    TORCH_CHECK(keep_current_indices.scalar_type() == torch::kLong, "keep_current_indices must have dtype torch.long");
    TORCH_CHECK(cache_tensor.device() == keep_current_indices.device(), "cache_tensor and keep_current_indices must be on the same device");

    const int64_t seq_dim = cache_tensor.dim() - 2;
    TORCH_CHECK(past_length >= 0, "past_length must be non-negative");
    TORCH_CHECK(past_length <= cache_tensor.size(seq_dim), "past_length exceeds cache sequence length");

    const int64_t current_length = cache_tensor.size(seq_dim) - past_length;
    if (current_length <= 0) {
        return cache_tensor;
    }

    const int64_t keep_count = keep_current_indices.numel();
    TORCH_CHECK(keep_count >= 0, "keep_count must be non-negative");
    TORCH_CHECK(keep_count <= current_length, "keep_count exceeds appended window length");

    if (keep_count == 0 || keep_count == current_length) {
        return cache_tensor;
    }

    auto tail = cache_tensor.narrow(seq_dim, past_length, current_length);
    auto kept_tail = tail.index_select(seq_dim, keep_current_indices);
    cache_tensor.narrow(seq_dim, past_length, keep_count).copy_(kept_tail);
    return cache_tensor;
}
"""
    try:
        module = load_inline(
            name="ddtree_compact_tail_ext_v1",
            cpp_sources=[cpp_source],
            functions=["compact_tail_inplace"],
            extra_cflags=["-O3"],
            verbose=False,
        )
        logger.info("Loaded inline C++ tail cache compaction extension for DDTree.")
        return module
    except Exception as exc:
        logger.warning(
            f"Failed to build inline C++ tail cache compaction extension; falling back to Python implementation. {exc}"
        )
        return None


def maybe_enable_cpp_compact(enabled: bool) -> None:
    global _CPP_COMPACT_ENABLED
    _CPP_COMPACT_ENABLED = enabled
    if enabled:
        load_cpp_compact_module()


def build_ddtree_tree(
    draft_logits: torch.Tensor,
    budget: int,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[dict[int, int]], torch.Tensor, dict[str, float]]:
    build_subtimes = empty_stage_times(DDTREE_TREE_BUILD_STAGE_ORDER)

    if budget <= 0 or draft_logits.shape[0] == 0:
        visibility = torch.zeros((1, 1), dtype=torch.bool)
        visibility[0, 0] = True
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            [-1],
            [dict()],
            visibility,
            build_subtimes,
        )

    topk = min(budget, draft_logits.shape[-1])
    depth_limit = int(draft_logits.shape[0])

    copy_start = cuda_time()
    logits = draft_logits.float()
    top_logits, top_token_ids = torch.topk(logits, k=topk, dim=-1)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    top_log_probs_cpu = (top_logits - log_z).to(device="cpu", dtype=torch.float32)
    top_token_ids_cpu = top_token_ids.to(device="cpu", dtype=torch.long)
    build_subtimes["tree_build_copy"] = cuda_time() - copy_start

    top_log_probs_np = top_log_probs_cpu.numpy()
    top_token_ids_np = top_token_ids_cpu.numpy()

    heap_start = time.perf_counter()
    first_logw = float(top_log_probs_np[0, 0])
    heap: list[tuple[float, tuple[int, ...], int, int, int, float]] = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    node_token_ids_np = np.empty(budget, dtype=np.int64)
    node_depths_np = np.empty(budget, dtype=np.int64)
    parents_np = np.empty(budget + 1, dtype=np.int32)
    parents_np[0] = -1
    child_maps: list[dict[int, int]] = [dict()]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids_np[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_np[node_count] = token_id
        node_depths_np[node_count] = depth
        parents_np[current_index] = parent_index
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_ranks = ranks[:-1] + (rank + 1,)
            sibling_logw = logw - float(top_log_probs_np[depth - 1, rank]) + float(top_log_probs_np[depth - 1, rank + 1])
            heapq.heappush(heap, (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw))

        if depth < depth_limit:
            child_ranks = ranks + (0,)
            child_logw = logw + float(top_log_probs_np[depth, 0])
            heapq.heappush(heap, (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw))

    build_subtimes["tree_build_heap"] = time.perf_counter() - heap_start

    visibility_start = time.perf_counter()
    current_length = 1 + node_count
    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = int(parents_np[index])
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True
    build_subtimes["tree_build_visibility"] = time.perf_counter() - visibility_start

    node_token_ids = torch.from_numpy(node_token_ids_np[:node_count])
    node_depths = torch.from_numpy(node_depths_np[:node_count])
    visibility = torch.from_numpy(visibility_np)
    parents = parents_np[:current_length].tolist()

    return node_token_ids, node_depths, parents, child_maps, visibility, build_subtimes


def compile_ddtree_tree(
    root_token_id: torch.Tensor,
    start: int,
    node_token_ids: torch.Tensor,
    node_depths: torch.Tensor,
    visibility_cpu: torch.Tensor,
    past_length: int,
    dtype: torch.dtype,
    device: torch.device,
    verify_input_ids_buffer: torch.Tensor,
    verify_position_ids_buffer: torch.Tensor,
    attention_mask_buffer: torch.Tensor,
    tree_visibility_buffer: torch.Tensor,
    previous_tree_start: int,
    previous_tree_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    current_length = 1 + int(node_token_ids.numel())

    if previous_tree_length > 0:
        attention_mask_buffer[0, 0, :previous_tree_length, previous_tree_start : previous_tree_start + previous_tree_length] = 0

    verify_input_ids = verify_input_ids_buffer[:, :current_length]
    verify_input_ids[0, 0] = root_token_id
    if current_length > 1:
        verify_input_ids[0, 1:current_length].copy_(node_token_ids, non_blocking=False)

    verify_position_ids = verify_position_ids_buffer[:, :current_length]
    verify_position_ids[0, 0] = start
    if current_length > 1:
        verify_position_ids[0, 1:current_length].copy_(node_depths, non_blocking=False)
        verify_position_ids[0, 1:current_length].add_(start)

    visibility = tree_visibility_buffer[:current_length, :current_length]
    visibility.copy_(visibility_cpu, non_blocking=False)

    tree_block = attention_mask_buffer[0, 0, :current_length, past_length : past_length + current_length]
    tree_block.fill_(torch.finfo(dtype).min)
    tree_block.masked_fill_(visibility, 0)

    attention_mask = attention_mask_buffer[:, :, :current_length, : past_length + current_length]
    return verify_input_ids, verify_position_ids, attention_mask, past_length, current_length


def follow_verified_tree(child_maps: list[dict[int, int]], posterior: torch.Tensor) -> tuple[list[int], int]:
    posterior_tokens = posterior[0].tolist()
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token


def _compact_appended_window(cache_tensor: torch.Tensor, past_length: int, keep_current_indices: torch.Tensor) -> None:
    current_length = cache_tensor.shape[-2] - past_length
    if current_length <= 0:
        return

    keep_count = keep_current_indices.numel()
    if keep_count == 0 or keep_count == current_length:
        return

    if _CPP_COMPACT_ENABLED:
        module = load_cpp_compact_module()
        if module is not None:
            module.compact_tail_inplace(cache_tensor, past_length, keep_current_indices)
            return

    kept_tail = cache_tensor.narrow(-2, past_length, current_length).index_select(-2, keep_current_indices)
    cache_tensor.narrow(-2, past_length, keep_count).copy_(kept_tail)


def compact_dynamic_cache(past_key_values: DynamicCache, past_length: int, keep_current_indices: list[int]) -> None:
    if len(keep_current_indices) == 0:
        past_key_values.crop(past_length)
        return

    keep_tensor_by_device: dict[torch.device, torch.Tensor] = {}

    def get_keep_tensor(device: torch.device) -> torch.Tensor:
        if device not in keep_tensor_by_device:
            keep_tensor_by_device[device] = torch.tensor(keep_current_indices, dtype=torch.long, device=device)
        return keep_tensor_by_device[device]

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        for layer_idx in range(len(past_key_values.key_cache)):
            key_cache = past_key_values.key_cache[layer_idx]
            value_cache = past_key_values.value_cache[layer_idx]
            keep_tensor = get_keep_tensor(key_cache.device)
            _compact_appended_window(key_cache, past_length, keep_tensor)
            _compact_appended_window(value_cache, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            if not hasattr(layer, "keys") or layer.keys is None or layer.keys.numel() == 0:
                continue
            keep_tensor = get_keep_tensor(layer.keys.device)
            _compact_appended_window(layer.keys, past_length, keep_tensor)
            _compact_appended_window(layer.values, past_length, keep_tensor)
        past_key_values.crop(past_length + len(keep_current_indices))
        return

    raise RuntimeError("Unsupported DynamicCache layout for DDTree cache compaction.")


@torch.inference_mode()
def ddtree_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    tree_budget: int | None = None,
    save_tree_traces: bool = False,
) -> SimpleNamespace:
    if block_size <= 1:
        return dflash_generate(
            model=model,
            target=target,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens
    draft_horizon = block_size - 1
    tree_budget = draft_horizon if tree_budget is None else max(tree_budget, 0)
    max_tree_nodes = 1 + tree_budget

    output_ids = torch.full(
        (1, max_length + max_tree_nodes),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    stop_token_ids_tensor = None if stop_token_ids is None else torch.tensor(stop_token_ids, device=model.device)

    verify_input_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    verify_position_ids_buffer = torch.empty((1, max_tree_nodes), dtype=torch.long, device=model.device)
    attention_mask_buffer = torch.zeros(
        (1, 1, max_tree_nodes, max_length + max_tree_nodes),
        dtype=target.dtype,
        device=model.device,
    )
    tree_visibility_buffer = torch.empty((max_tree_nodes, max_tree_nodes), dtype=torch.bool, device=model.device)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()
    stage_times = empty_stage_times(DDTREE_STAGE_ORDER + DDTREE_TREE_BUILD_STAGE_ORDER)

    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    round_clock_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    round_timestamps = []
    round_trees = [] if save_tree_traces else None
    draft_prefill = True
    previous_tree_start = 0
    previous_tree_length = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        root_token = block_output_ids[:, :1]

        draft_stage_start = cuda_time()
        noise_embedding = target.model.embed_tokens(block_output_ids)
        draft_logits = target.lm_head(model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, -draft_horizon:, :])
        past_key_values_draft.crop(start)
        draft_stage_elapsed = cuda_time() - draft_stage_start
        if draft_prefill:
            draft_prefill = False
            decode_start = cuda_time()
        else:
            stage_times["draft"] += draft_stage_elapsed

        tree_build_start = cuda_time()
        node_token_ids, node_depths, parents, child_maps, visibility_cpu, tree_build_subtimes = build_ddtree_tree(
            draft_logits[0], tree_budget
        )
        stage_times["tree_build"] += cuda_time() - tree_build_start
        for stage_name, stage_elapsed in tree_build_subtimes.items():
            stage_times[stage_name] += stage_elapsed

        tree_compile_start = cuda_time()
        verify_input_ids, verify_position_ids, verify_attention_mask, previous_tree_start, previous_tree_length = compile_ddtree_tree(
            root_token_id=root_token[0, 0],
            start=start,
            node_token_ids=node_token_ids,
            node_depths=node_depths,
            visibility_cpu=visibility_cpu,
            past_length=start,
            dtype=target.dtype,
            device=model.device,
            verify_input_ids_buffer=verify_input_ids_buffer,
            verify_position_ids_buffer=verify_position_ids_buffer,
            attention_mask_buffer=attention_mask_buffer,
            tree_visibility_buffer=tree_visibility_buffer,
            previous_tree_start=previous_tree_start,
            previous_tree_length=previous_tree_length,
        )
        stage_times["tree_compile"] += cuda_time() - tree_compile_start

        verify_stage_start = cuda_time()
        output = target(
            verify_input_ids,
            position_ids=verify_position_ids,
            attention_mask=verify_attention_mask,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        stage_times["verify"] += cuda_time() - verify_stage_start

        commit_stage_start = cuda_time()
        posterior = sample(output.logits, temperature)
        accepted_indices, next_token = follow_verified_tree(child_maps, posterior)
        accepted_index_tensor = torch.tensor(accepted_indices, dtype=torch.long, device=verify_input_ids.device)
        accepted_tokens = verify_input_ids.index_select(1, accepted_index_tensor)

        output_ids[:, start : start + len(accepted_indices)] = accepted_tokens
        output_ids[:, start + len(accepted_indices)] = next_token

        compact_dynamic_cache(past_key_values_target, start, accepted_indices)
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids).index_select(1, accepted_index_tensor)

        acceptance_lengths.append(len(accepted_indices))
        start += len(accepted_indices)
        stage_times["commit"] += cuda_time() - commit_stage_start
        round_timestamps.append(cuda_time() - round_clock_start)
        if save_tree_traces:
            round_trees.append({
                "accepted_indices": [int(index) for index in accepted_indices],
                "tree": {
                    "node_token_ids": [int(token_id) for token_id in node_token_ids.tolist()],
                    "node_depths": [int(depth) for depth in node_depths.tolist()],
                    "parents": [int(parent) for parent in parents],
                },
            })

        if stop_token_ids_tensor is not None:
            new_tokens = output_ids[:, start - len(accepted_indices) : start + 1]
            if torch.isin(new_tokens[0], stop_token_ids_tensor).any():
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids_tensor is not None:
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_tensor).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids.cpu(),
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        decode_rounds=len(acceptance_lengths),
        stage_times=stage_times,
        round_timestamps=round_timestamps,
        round_trees=round_trees,
    )
