import torch
import math
from typing import Type, Dict, Any, Tuple, Callable

from . import merge
from .utils import isinstance_str, init_generator
from causvid.models.wan.causal_model import flash_attn_interface



def compute_merge(x: torch.Tensor, tome_info: Dict[str, Any]) -> Tuple[Callable, ...]:
    args = tome_info["args"]

    r = int(x.shape[1] * args["ratio"])

    # Re-init the generator if it hasn't already been initialized or device has changed.
    if args["generator"] is None:
        args["generator"] = init_generator(x.device)
    elif args["generator"].device != x.device:
        args["generator"] = init_generator(x.device, fallback=args["generator"])
    
    B, L, N, D = x.shape
    y = x.reshape(B, L, N * D)   # [B,L,C]
    m, u = merge.bipartite_soft_matching(y, r)

    m_a, u_a = (m, u) if args["merge_attn"]      else (merge.do_nothing, merge.do_nothing)
    m_c, u_c = (m, u) if args["merge_crossattn"] else (merge.do_nothing, merge.do_nothing)
    m_m, u_m = (m, u) if args["merge_mlp"]       else (merge.do_nothing, merge.do_nothing)

    return m_a, m_c, m_m, u_a, u_c, u_m  # Okay this is probably not very good


def make_diffusers_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """
    class ToMeBlock(block_class):
        # Save for unpatching later
        _parent = block_class

        def forward(
            self,
            hidden_states,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            timestep=None,
            cross_attention_kwargs=None,
            class_labels=None,
        ) -> torch.Tensor:
            # (1) ToMe
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(hidden_states, self._tome_info)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) ToMe m_a
            norm_hidden_states = m_a(norm_hidden_states)

            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) ToMe u_a
            hidden_states = u_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) ToMe m_c
                norm_hidden_states = m_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) ToMe u_c
                hidden_states = u_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)
            
            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) ToMe m_m
            norm_hidden_states = m_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) ToMe u_m
            hidden_states = u_m(ff_output) + hidden_states

            return hidden_states

    return ToMeBlock


def make_causal_wan_sa_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a causal Wan attention block.
    """
    class ToMeBlock(block_class):
        _parent = block_class

        def _merge_4d(self, x, merge_fn):
            # x: [B, L, N, D]  -> merge along L
            B, L, N, D = x.shape
            y = x.reshape(B, L, N * D)   # [B,L,C]
            y = merge_fn(y)              # [B,L',C]
            Lp = y.shape[1]
            return y.reshape(B, Lp, N, D)

        def _unmerge_4d(self, x, unmerge_fn):
            B, Lp, N, D = x.shape
            y = x.reshape(B, Lp, N * D)  # [B,L',C]
            y = unmerge_fn(y)            # [B,L,C]
            L  = y.shape[1]
            return y.reshape(B, L, N, D)

        def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, current_end=0):
            r"""
            Args:
                x(Tensor): Shape [B, L, num_heads, C / num_heads]
                seq_lens(Tensor): Shape [B]
                grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
                freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
                block_mask (BlockMask)
            """
            b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

            # query, key, value function
            def qkv_fn(x):
                q = self.norm_q(self.q(x)).view(b, s, n, d)
                k = self.norm_k(self.k(x)).view(b, s, n, d)
                v = self.v(x).view(b, s, n, d)
                return q, k, v

            q, k, v = qkv_fn(x)

            if kv_cache is None:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                    torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
            else:
                frame_seqlen = math.prod(grid_sizes[0][1:]).item()
                current_start_frame = current_start // frame_seqlen
                roped_query = self.causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key = self.causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

                seq_lens = []
                for i, c_start in enumerate(current_start):
                    current_end = c_start + roped_query.shape[1]
                    sink_tokens = self.sink_size * frame_seqlen
                    # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
                    kv_cache_size = kv_cache["k"].shape[1]
                    num_new_tokens = roped_query.shape[1]
                    if c_start + num_new_tokens >= kv_cache_size:
                        kv_cache["global_end_index"][i].fill_(c_start)
                        kv_cache["local_end_index"][i].fill_(kv_cache_size)
                    if (current_end > kv_cache["global_end_index"][i].item()) and (
                            num_new_tokens + kv_cache["local_end_index"][i].item() > kv_cache_size):
                        # Calculate the number of new tokens added in this step
                        # Shift existing cache content left to discard oldest tokens
                        # Clone the source slice to avoid overlapping memory error
                        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"][i].item() - kv_cache_size
                        num_rolled_tokens = kv_cache["local_end_index"][i].item() - num_evicted_tokens - sink_tokens
                        kv_cache["k"][i:i+1, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache["k"][i:i+1, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        kv_cache["v"][i:i+1, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache["v"][i:i+1, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        # Insert the new keys/values at the end
                        local_end_index = kv_cache["local_end_index"][i].item() + current_end - \
                            kv_cache["global_end_index"][i].item() - num_evicted_tokens
                    else:
                        local_end_index = kv_cache["local_end_index"][i].item() + current_end - kv_cache["global_end_index"][i].item()

                    local_start_index = local_end_index - num_new_tokens
                    kv_cache["k"][i:i+1, local_start_index:local_end_index] = roped_key[i:i+1]
                    kv_cache["v"][i:i+1, local_start_index:local_end_index] = v[i:i+1]

                    seq_lens.append(local_end_index)

                    kv_cache["global_end_index"][i].fill_(current_end)
                    kv_cache["local_end_index"][i].fill_(local_end_index)
                
                seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=roped_query.device)

                # Merge here
                m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(roped_query, self._tome_info)
                roped_query = self._merge_4d(roped_query, m_a)

                x = flash_attn_interface.flash_attn_with_kvcache(
                    q=roped_query,
                    k_cache=kv_cache["k"][:, :seq_lens.max()],
                    v_cache=kv_cache["v"][:, :seq_lens.max()],
                    cache_seqlens=seq_lens,
                )

                # unmerge here
                x = self._unmerge_4d(x, u_a)

            # output
            x = x.flatten(2)
            x = self.o(x)
            return x

    return ToMeBlock


def make_causal_wan_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a causal Wan attention block.
    """
    class ToMeBlock(block_class):
        _parent = block_class

        def forward(
            self,
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
            block_mask,
            kv_cache=None,
            crossattn_cache=None,
            current_start=0,
            current_end=0
        ):
            r"""
            Args:
                x(Tensor): Shape [B, L, C]
                e(Tensor): Shape [B, F, 6, C]
                seq_lens(Tensor): Shape [B], length of each sequence in batch
                grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
                freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            """
            m_a, m_c, m_m, u_a, u_c, u_m = compute_merge(x, self._tome_info)

            num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
            # assert e.dtype == torch.float32
            # with amp.autocast(dtype=torch.float32):
            e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
            # assert e[0].dtype == torch.float32

            # self-attention
            norm_x = self.norm1(x)
            norm_x = norm_x.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            norm_x = (norm_x * (1 + e[1]) + e[0]).flatten(1, 2)

            # ToMe merge before attention
            norm_x = m_a(norm_x)

            y = self.self_attn(
                norm_x,
                seq_lens,
                grid_sizes,
                freqs,
                block_mask,
                kv_cache,
                current_start,
                current_end
            )

            # Unmerge after attention
            y = u_a(y)

            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                    * e[2]).flatten(1, 2)

            # cross-attention & ffn function
            norm_x = self.norm3(x)
            norm_x = m_c(norm_x)  # ToMe merge before cross-attn

            x = x + u_c(
                self.cross_attn(norm_x, context, context_lens, crossattn_cache=crossattn_cache)
            )
            
            # ffn
            norm_x = self.norm2(x)
            norm_x = norm_x.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
            norm_x = (norm_x * (1 + e[4]) + e[3]).flatten(1, 2)
            norm_x = m_m(norm_x)  # ToMe merge before FFN

            y = self.ffn(norm_x)
            y = u_m(y)  # ToMe unmerge after FFN

            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                        frame_seqlen)) * e[5]).flatten(1, 2)
            return x

    return ToMeBlock


def hook_tome_model(model: torch.nn.Module, is_dit=False):
    """ Adds a forward pre hook to get the image size. This hook can be removed with remove_patch. """
    def hook(module, args):
        token_size = (args[0].shape[2], args[0].shape[3]) if is_dit is False else args[0].shape[1]
        module._tome_info["size"] = token_size
        return None

    model._tome_info["hooks"].append(model.register_forward_pre_hook(hook))


def apply_patch(
        diffusion_model: torch.nn.Module,
        ratio: float = 0.2,
        merge_attn: bool = True,
        merge_crossattn: bool = False,
        merge_mlp: bool = False):
    """
    Patches a stable diffusion model with ToMe.
    Apply this to the highest level stable diffusion object (i.e., it should have a .model.diffusion_model).

    Important Args:
     - diffusion_model: CausalWanModel
     - ratio: The ratio of tokens to merge. I.e., 0.4 would reduce the total number of tokens by 40%.
              The maximum value for this is 1-(1/(sx*sy)). By default, the max is 0.75 (I recommend <= 0.5 though).
              Higher values result in more speed-up, but with more visual quality loss.
    
    Args to tinker with if you want:
     - max_downsample [1, 2, 4, or 8]: Apply ToMe to layers with at most this amount of downsampling.
                                       E.g., 1 only applies to layers with no downsampling (4/15) while
                                       8 applies to all layers (15/15). I recommend a value of 1 or 2.
     - sx, sy: The stride for computing dst sets (see paper). A higher stride means you can merge more tokens,
               but the default of (2, 2) works well in most cases. Doesn't have to divide image size.
     - use_rand: Whether or not to allow random perturbations when computing dst sets (see paper). Usually
                 you'd want to leave this on, but if you're having weird artifacts try turning this off.
     - merge_attn: Whether or not to merge tokens for attention (recommended).
     - merge_crossattn: Whether or not to merge tokens for cross attention (not recommended).
     - merge_mlp: Whether or not to merge tokens for the mlp layers (very not recommended).
    """

    # Make sure the module is not currently patched
    remove_patch(diffusion_model)

    diffusion_model._tome_info = {
        "size": None,
        "hooks": [],
        "args": {
            "ratio": ratio,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp
        }
    }
    hook_tome_model(diffusion_model, is_dit=True)

    for module in diffusion_model.blocks:
        module_sa = module.self_attn
        # If for some reason this has a different name, create an issue and I'll fix it
        if isinstance_str(module_sa, "CausalWanSelfAttention"):
            module_sa.__class__  = make_causal_wan_sa_tome_block(module_sa.__class__)
            module_sa._tome_info = diffusion_model._tome_info

    return diffusion_model


def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a ToMe Diffusion module if it was already patched. """
    # For diffusers
    model = model.unet if hasattr(model, "unet") else model

    for _, module in model.named_modules():
        if hasattr(module, "_tome_info"):
            for hook in module._tome_info["hooks"]:
                hook.remove()
            module._tome_info["hooks"].clear()

        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent
    
    return model