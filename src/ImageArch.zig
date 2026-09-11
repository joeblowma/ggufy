const std = @import("std");
const types = @import("types.zig");

/// A single dimension constraint, used to tell apart architectures whose tensor
/// *names* are identical (Mage-Flow vs Qwen-Image). Mirrors what ComfyUI's
/// model_detection.py does in the same situation: read one dimension of one
/// named tensor and compare it against a known constant.
pub const ShapeRule = struct {
    /// Tensor name with any state-dict prefix already stripped (see stripPrefix).
    key: []const u8,
    /// Index into `Tensor.dims`, which is always outermost-first (PyTorch order)
    /// regardless of whether the source was SafeTensors or GGUF.
    dim: usize,
    /// Required extent of that dimension.
    extent: usize,
};

/// A key that only needs protecting when its rows are short. Some checkpoints ship
/// the same layer either full-width or projected onto a small basis; the wide form
/// quantizes fine, while the narrow form would put many unrelated rows in one block.
pub const NarrowRule = struct {
    /// Tensor name substring, matched the same way `keys_hiprec` is.
    key: []const u8,
    /// Protect when the contiguous (last) dimension is below this.
    below_cols: usize,
};

/// Minimum precision a tensor may be stored at, as a bits-per-weight class.
/// Deliberately not a concrete type: an architecture knows a layer needs more
/// bits, not which of the six 4-bit formats someone will ask for. The output
/// family is inherited from the requested type (see Convert.liftToFloor).
pub const Precision = enum(u8) {
    bits4 = 4,
    bits5 = 5,
    bits6 = 6,
    bits8 = 8,
    bits16 = 16,
};

/// A tensor that breaks below some precision, whatever format is requested.
/// Unlike `keys_hiprec` this does not jump to the source dtype - it lifts to the
/// cheapest rung of the requested format's own family that clears the floor.
pub const PrecisionFloor = struct {
    /// Tensor name substring, matched the same way `keys_hiprec` is.
    key: []const u8,
    min: Precision,
};

/// Represents a model architecture with its detection keys and configuration
pub const Arch = struct {
    /// String describing architecture name
    name: []const u8,
    /// Whether to reshape tensors for this architecture
    shape_fix: bool = false,
    /// List of key sets to match in state dict (any set matching = detected)
    /// Each inner slice is a set of keys that must ALL be present
    keys_detect: []const []const []const u8,
    /// Keys that mark model as invalid for conversion (e.g., wrong format)
    keys_banned: []const []const u8 = &.{},
    /// Dimension constraints that must ALL hold in addition to `keys_detect`.
    /// Only usable when tensor shapes are available, so an architecture that
    /// declares these can only be detected via the tensor-based entry points
    /// (`detectArchFromTensors*`), never from a bare list of names.
    shape_detect: []const ShapeRule = &.{},
    /// Keys that need to be kept in fp32/high precision
    keys_hiprec: []const []const u8 = &.{},
    /// Keys kept high-precision only in their narrow form (see NarrowRule)
    keys_hiprec_narrow: []const NarrowRule = &.{},
    /// Tensors that must not be stored below a given precision (see PrecisionFloor).
    /// Applied to the target type before any format branch, so it holds on every
    /// output path and is unaffected by `-a` or `-x`.
    precision_floor: []const PrecisionFloor = &.{},
    /// Key substrings to ignore when found
    keys_ignore: []const []const u8 = &.{},
    /// Quantization threshhold specific to a model, or fall back to default
    threshhold: ?u64,
    /// Sensitivities filename; json dictionary of layer names and their relative sensitivity to quantization, 1-100
    sensitivities: []const u8 = "",
    /// Keys that should be upcast from bf16 to fp32 (start with a dot for end match)
    upcast_from_bf16: []const []const u8 = &.{},
    /// Keys that must pass through as-is in NVFP4 output (ComfyUI reads their shape[1] for arch detection)
    keys_nvfp4_passthrough: []const []const u8 = &.{},
    /// JSON object with base architecture configs (e.g. vae/audio_vae/vocoder) that may be absent
    /// from fine-tuned source files. Top-level keys are merged into the output `config` KV,
    /// with the source file's keys taking priority over these defaults.
    base_config_json: []const u8 = "",

    /// Check if this architecture matches the given tensor names
    pub fn matches(self: Arch, tensor_names: []const []const u8) bool {
        for (self.keys_detect) |key_set| {
            if (allKeysPresent(key_set, tensor_names)) {
                // Check if any banned keys are present; if so, skip this key set
                var banned = false;
                for (self.keys_banned) |banned_key| {
                    if (containsKey(tensor_names, banned_key)) {
                        std.log.debug("Skipping key set for architecture {s}: found banned key {s}", .{ self.name, banned_key });
                        banned = true;
                        break;
                    }
                }
                if (banned) continue;
                return true;
            }
        }
        return false;
    }

    /// Check this architecture's `shape_detect` constraints against a tensor list.
    /// A missing tensor, a missing dimension, or a mismatched extent all fail.
    pub fn shapesMatch(self: Arch, tensors: []const types.Tensor) bool {
        for (self.shape_detect) |rule| {
            const t = findTensor(tensors, rule.key) orelse return false;
            if (rule.dim >= t.dims.len) return false;
            if (t.dims[rule.dim] != rule.extent) return false;
        }
        return true;
    }

    /// Check if a key should be kept in high precision
    pub fn isHighPrecision(self: Arch, key: []const u8) bool {
        for (self.keys_hiprec) |hiprec| {
            if (std.mem.indexOf(u8, key, hiprec) != null) {
                return true;
            }
        }
        return false;
    }

    /// Check if a key should be kept in high precision given the shape it arrived with.
    /// `dims` is outermost-first, so the contiguous extent is the last entry.
    pub fn isNarrowHighPrecision(self: Arch, key: []const u8, dims: []const usize) bool {
        if (dims.len == 0) return false;
        const cols = dims[dims.len - 1];
        for (self.keys_hiprec_narrow) |rule| {
            if (cols < rule.below_cols and std.mem.indexOf(u8, key, rule.key) != null) return true;
        }
        return false;
    }

    /// The strictest precision floor declared for this key, or null if none apply.
    pub fn precisionFloor(self: Arch, key: []const u8) ?Precision {
        var strictest: ?Precision = null;
        for (self.precision_floor) |rule| {
            if (std.mem.indexOf(u8, key, rule.key) == null) continue;
            if (strictest == null or @intFromEnum(rule.min) > @intFromEnum(strictest.?)) {
                strictest = rule.min;
            }
        }
        return strictest;
    }

    /// Check if a key should be ignored
    pub fn shouldIgnore(self: Arch, key: []const u8) bool {
        for (self.keys_ignore) |ignore| {
            if (std.mem.indexOf(u8, key, ignore) != null) {
                return true;
            }
        }
        return false;
    }

    /// Check if any of the given tensor names are banned for this architecture
    /// Returns the first banned key found, or null if none are banned
    pub fn findBannedKey(self: Arch, tensor_names: []const []const u8) ?[]const u8 {
        for (self.keys_banned) |banned| {
            if (containsKey(tensor_names, banned)) {
                return banned;
            }
        }
        return null;
    }

    /// Check if any of the given tensor names are banned (returns bool)
    pub fn hasBannedKeys(self: Arch, tensor_names: []const []const u8) bool {
        return self.findBannedKey(tensor_names) != null;
    }

    /// Check if a key must pass through unquantized in NVFP4 output for ComfyUI compat
    pub fn isNvfp4Passthrough(self: Arch, key: []const u8) bool {
        for (self.keys_nvfp4_passthrough) |pattern| {
            if (std.mem.indexOf(u8, key, pattern) != null) return true;
        }
        return false;
    }

    /// Check if the key should be upcast from bf16
    pub fn shouldUpcast(self: Arch, tensor_name: []const u8) bool {
        for (self.upcast_from_bf16) |pattern| {
            if (pattern.len > 0 and pattern[0] == '.') {
                // Dot-prefixed: match if tensor_name ends with this pattern
                if (std.mem.endsWith(u8, tensor_name, pattern)) return true;
            } else {
                // No dot: match if tensor_name equals this pattern
                if (std.mem.eql(u8, tensor_name, pattern)) return true;
            }
        }
        return false;
    }
};

fn allKeysPresent(key_set: []const []const u8, tensor_names: []const []const u8) bool {
    for (key_set) |key| {
        if (!containsKey(tensor_names, key)) {
            return false;
        }
    }
    return true;
}

fn containsKey(tensor_names: []const []const u8, key: []const u8) bool {
    for (tensor_names) |name| {
        const stripped = stripPrefix(name);
        if (std.mem.eql(u8, stripped, key)) {
            return true;
        }
    }
    return false;
}

fn findTensor(tensors: []const types.Tensor, key: []const u8) ?*const types.Tensor {
    for (tensors) |*t| {
        if (std.mem.eql(u8, stripPrefix(t.name), key)) return t;
    }
    return null;
}

/// Check if tensors contain any banned keys for a specific architecture
/// Returns the first banned key found, or null if none are banned
pub fn findBannedKeyInTensors(arch: *const Arch, tensors: []const types.Tensor) ?[]const u8 {
    var names: [4096][]const u8 = undefined;
    const count = @min(tensors.len, 4096);
    for (tensors[0..count], 0..) |t, i| {
        names[i] = t.name;
    }
    return arch.findBannedKey(names[0..count]);
}

/// Check if tensors contain any banned keys for a specific architecture (returns bool)
pub fn hasBannedKeysInTensors(arch: *const Arch, tensors: []const types.Tensor) bool {
    return findBannedKeyInTensors(arch, tensors) != null;
}

// ============================================================================
// Architecture Definitions
// ============================================================================

pub const flux = Arch{
    .name = "flux",
    .shape_fix = true,
    .keys_detect = &.{
        &.{"transformer_blocks.0.attn.norm_added_k.weight"},
        &.{"double_blocks.0.img_attn.proj.weight"},
    },
    .keys_banned = &.{"transformer_blocks.0.attn.norm_added_k.weight"},
    .threshhold = null,
    .upcast_from_bf16 = &.{
        ".norm.query_norm.scale",
        ".norm.key_norm.scale",
        ".norm.query_norm.weight",
        ".norm.key_norm.weight",
    },
    // ComfyUI infers in_channels from img_in.weight.shape[1], context_in_dim from
    // txt_in.weight.shape[1], and vec_in_dim from vector_in.in_layer.weight.shape[1].
    // NVFP4 nibble-packing halves the column count, so ComfyUI detects half the true
    // dimension and then clips the dequantized weight, causing shape mismatches at runtime.
    // Keep these as BF16 so ComfyUI reads the correct dimensions.
    .keys_nvfp4_passthrough = &.{
        "img_in.weight",
        "txt_in.weight",
        "vector_in.in_layer.weight",
    },
};

pub const sd3 = Arch{
    .name = "sd3",
    .keys_detect = &.{
        &.{"transformer_blocks.0.attn.add_q_proj.weight"},
        &.{"joint_blocks.0.x_block.attn.qkv.weight"},
    },
    .keys_banned = &.{"transformer_blocks.0.attn.add_q_proj.weight"},
    .threshhold = null,
    // ComfyUI infers adm_in_channels from y_embedder.mlp.0.weight.shape[1] and
    // context_dim from context_embedder.weight.shape[1]; NVFP4 packing halves both.
    .keys_nvfp4_passthrough = &.{
        "y_embedder.mlp.0.weight",
        "context_embedder.weight",
    },
};

pub const aura = Arch{
    .name = "aura",
    .keys_detect = &.{
        &.{"double_layers.3.modX.1.weight"},
        &.{"joint_transformer_blocks.3.ff_context.out_projection.weight"},
    },
    .keys_banned = &.{"joint_transformer_blocks.3.ff_context.out_projection.weight"},
    .threshhold = null,
};

pub const hidream = Arch{
    .name = "hidream",
    .keys_detect = &.{
        &.{
            "caption_projection.0.linear.weight",
            "double_stream_blocks.0.block.ff_i.shared_experts.w3.weight",
        },
    },
    .keys_hiprec = &.{
        ".ff_i.gate.weight",
        "img_emb.emb_pos",
    },
    .threshhold = null,
};

// Anima is Cosmos-Predict2 (MiniTrainDIT) with an extra bolted-on T5 text
// adapter (`llm_adapter`). It shares Cosmos's entire backbone, so its detect
// keys are Cosmos's two plus the llm_adapter discriminator. This mirrors
// ComfyUI's own model_detection.py, which starts at "cosmos_predict2" and
// reclassifies to "anima" iff `llm_adapter.blocks.0.cross_attn.q_proj.weight`
// is present. "anima" is a valid `general.architecture` value for the
// ComfyUI-GGUF loader (it's in PIG_ARCH_LIST), so we can name it distinctly.
//
// Must be listed BEFORE `cosmos` in arch_list: base Cosmos's key set is a
// subset of Anima's, so cosmos would otherwise match first.
//
// The ENTIRE `llm_adapter` is kept high-precision (not just its embedding),
// matching the reference converter silveroxides/convert_to_quant (its
// ANIMA_LAYER_KEYNAMES lists "llm_adapter" as highprec). Two reasons:
//   1. ComfyUI: the adapter's `embed.weight` is an nn.Embedding table that
//      can't be block/int-quantized (also caught generically by
//      isEmbeddingWeight() in Convert.zig).
//   2. Forge Neo: its loader's `process_anima` MOVES the whole llm_adapter out
//      of the transformer and into the *text-encoder* component. If any adapter
//      tensor is quantized (carries `.comfy_quant`), Forge loads the text
//      encoder via its MixedPrecision path, which builds non-quantized layers
//      (the `embed`) at fp32 while the quantized projections dequantize to
//      bf16. The adapter then computes rotary embeddings from the fp32 embed
//      output and applies them to q/k, but v (no rope) stays bf16 — so
//      scaled_dot_product_attention gets mismatched dtypes and throws. Keeping
//      the adapter fully bf16 means the text encoder has no `.comfy_quant`, so
//      it loads in plain bf16 and everything matches. (bf16 Linears load fine
//      in ComfyUI too, so this does not regress the working ComfyUI path.)
pub const anima = Arch{
    .name = "anima",
    .keys_detect = &.{
        &.{
            "blocks.0.mlp.layer1.weight",
            "blocks.0.adaln_modulation_cross_attn.1.weight",
            "llm_adapter.blocks.0.cross_attn.q_proj.weight",
        },
    },
    // High-precision set mirrors silveroxides/convert_to_quant's ANIMA_LAYER_KEYNAMES
    // (the reference converter): the llm_adapter (see above), plus the first block,
    // block 1's adaln modulation, the final layer, and the timestep/patch embedders.
    // These are the small, sensitivity-critical layers that reference tool keeps in
    // full precision for quality. Patterns are bare substrings (isHighPrecision matches
    // the full tensor name), so "blocks.0." also covers the — already-hiprec — llm_adapter
    // block 0, which is harmless. pos_embedder is retained from the Cosmos base.
    .keys_hiprec = &.{
        "pos_embedder",
        "llm_adapter",
        "blocks.0.",
        "blocks.1.adaln_modulation",
        "final_layer",
        "t_embedder",
        "x_embedder",
    },
    .keys_ignore = &.{ "_extra_state", "accum_" },
    .threshhold = null,
};

pub const cosmos = Arch{
    .name = "cosmos",
    .keys_detect = &.{
        &.{
            "blocks.0.mlp.layer1.weight",
            "blocks.0.adaln_modulation_cross_attn.1.weight",
        },
    },
    .keys_hiprec = &.{"pos_embedder"},
    .keys_ignore = &.{ "_extra_state", "accum_" },
    .threshhold = null,
};

pub const hyvid = Arch{
    .name = "hyvid",
    .keys_detect = &.{
        &.{
            "double_blocks.0.img_attn_proj.weight",
            "txt_in.individual_token_refiner.blocks.1.self_attn_qkv.weight",
        },
    },
    .threshhold = null,
};

pub const wan = Arch{
    .name = "wan",
    .keys_detect = &.{
        &.{
            "blocks.0.self_attn.norm_q.weight",
            "text_embedding.2.weight",
            "head.modulation",
        },
    },
    .keys_hiprec = &.{".modulation"},
    .threshhold = null,
};

pub const ltxv = Arch{
    .name = "ltxv",
    .keys_detect = &.{
        &.{
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.27.scale_shift_table",
            "caption_projection.linear_2.weight",
        },
    },
    .keys_hiprec = &.{"scale_shift_table"},
    .threshhold = null,
};

pub const ltx2 = Arch{
    // ComfyUI identifies both v1 and 2.x by the "ltxv" architecture string.
    .name = "ltxv",
    .base_config_json = @embedFile("configs/ltx23_base_config.json"),
    .keys_detect = &.{
        &.{
            "adaln_single.emb.timestep_embedder.linear_2.weight",
            "transformer_blocks.47.scale_shift_table",
            "patchify_proj.weight",
        },
    },
    // Tensors that must stay in source precision:
    //   - scale_shift_table: conditioning signals (multiple variants in 2.x)
    //   - _norm.weight: RMSNorm scale vectors
    //   - .bias: bias vectors must not be block-quantized
    //   - adaln_single: AdaLN conditioning projections, sensitive and small outer-dim shapes
    //   - patchify_proj.weight / proj_out.weight: patch embed/unembed, outer-dim = 128
    //   - learnable_registers: embedding tokens [128, X] — Python shape[-1]=128, not divisible by Q4_K block size 256
    .keys_hiprec = &.{
        "scale_shift_table",
        "_norm.weight",
        ".bias",
        "adaln_single",
        "patchify_proj.weight",
        "proj_out.weight",
        "learnable_registers",
    },
    .threshhold = null,
};

pub const sdxl = Arch{
    .name = "sdxl",
    .shape_fix = true,
    .keys_detect = &.{
        &.{ "down_blocks.0.downsamplers.0.conv.weight", "add_embedding.linear_1.weight" },
        // Non-diffusers format
        &.{
            "input_blocks.3.0.op.weight",
            "input_blocks.6.0.op.weight",
            "output_blocks.2.2.conv.weight",
            "output_blocks.5.2.conv.weight",
        },
        &.{"label_emb.0.0.weight"},
    },
    .threshhold = null,
    .sensitivities = @embedFile("sensitivities/sdxl.json"),
    // ComfyUI infers adm_in_channels from label_emb.0.0.weight.shape[1]; NVFP4 packing halves it.
    .keys_nvfp4_passthrough = &.{
        "label_emb.0.0.weight",
    },
};

pub const sd1 = Arch{
    .name = "sd1",
    .shape_fix = true,
    .keys_detect = &.{
        &.{"down_blocks.0.downsamplers.0.conv.weight"},
        // Non-diffusers format
        &.{
            "input_blocks.3.0.op.weight",
            "input_blocks.6.0.op.weight",
            "input_blocks.9.0.op.weight",
            "output_blocks.2.1.conv.weight",
            "output_blocks.5.2.conv.weight",
            "output_blocks.8.2.conv.weight",
        },
    },
    .threshhold = null,
    .sensitivities = @embedFile("sensitivities/sd1.5.json"),
    // ComfyUI infers adm_in_channels from label_emb.0.0.weight.shape[1] on class-conditional
    // SD1 variants; NVFP4 packing halves it.
    .keys_nvfp4_passthrough = &.{
        "label_emb.0.0.weight",
    },
};

pub const lumina2 = Arch{
    .name = "lumina2",
    .keys_detect = &.{
        &.{ "cap_embedder.1.weight", "context_refiner.0.attention.qkv.weight" },
    },
    .shape_fix = true,
    .keys_ignore = &.{
        "norm_final.weight",
    },
    .threshhold = 8192,
    .upcast_from_bf16 = &.{
        "cap_pad_token",
        "x_pad_token",
    },
    // ComfyUI infers cap_feat_dim from cap_embedder.1.weight.shape[1]. NVFP4 nibble-packing
    // halves that dimension, causing a shape mismatch when loading. Keep as BF16 so ComfyUI
    // reads the correct dimension.
    .keys_nvfp4_passthrough = &.{
        "cap_embedder.1.weight",
    },
};

pub const qwen = Arch{
    // "qwen_image" is the image_model string ComfyUI assigns, and the GGUF loader
    // gates on it verbatim.
    .name = "qwen_image",
    .keys_detect = &.{
        &.{
            "time_text_embed.timestep_embedder.linear_2.weight",
            "transformer_blocks.0.attn.norm_added_q.weight",
            "transformer_blocks.0.img_mlp.net.0.proj.weight",
        },
    },
    .shape_fix = true,
    .threshhold = null,
    .upcast_from_bf16 = &.{
        "txt_norm.weight",
        ".norm_k.weight",
        ".norm_q.weight",
        ".norm_added_k.weight",
        ".norm_added_q.weight",
    },
    // ComfyUI infers in_channels from img_in.weight.shape[1]; NVFP4 packing halves it.
    .keys_nvfp4_passthrough = &.{
        "img_in.weight",
    },
};

// Mage-Flow (microsoft/Mage) is a 12-layer native-resolution MMDiT that reuses
// Qwen-Image's double-stream block verbatim. Its state dict has *exactly* the
// same set of tensor names as Qwen-Image — only the dimensions differ — so name
// matching alone cannot tell the two apart. ComfyUI's model_detection.py
// disambiguates purely by shape (txt_norm/proj_out are 2560/128 here vs
// 3584/64 for Qwen-Image), and so do we, via `shape_detect`.
//
// Must be listed BEFORE `qwen` in arch_list: Qwen-Image's key set matches
// Mage-Flow's file exactly, so qwen would otherwise win.
//
// `mage_flow` is the `image_model` string ComfyUI itself assigns, so we use it
// verbatim as `general.architecture`.
pub const mageflow = Arch{
    .name = "mage_flow",
    .keys_detect = &.{
        &.{
            "time_text_embed.timestep_embedder.linear_2.weight",
            "transformer_blocks.0.attn.norm_added_q.weight",
            "transformer_blocks.0.img_mlp.net.0.proj.weight",
            "txt_norm.weight",
            "proj_out.weight",
        },
    },
    // The exact pair ComfyUI reads to separate Mage-Flow from Qwen-Image.
    .shape_detect = &.{
        .{ .key = "txt_norm.weight", .dim = 0, .extent = 2560 },
        .{ .key = "proj_out.weight", .dim = 0, .extent = 128 },
    },
    .shape_fix = true,
    .threshhold = null,
    // Only 12 blocks and 4.1B params, so the conditioning/IO path is under 1%
    // of the weights (~38M params) while carrying most of the quantization
    // risk. It lands as F32 in GGUF output, which costs ~144 MiB — about 6% of
    // a Q4_K build. Same trade-off the krea2/ltx2 entries above make.
    //   - txt_norm.weight: also load-bearing for detection. It is 1-D, which
    //     already protects it in GGUF output, but the SafeTensors cluster
    //     formats (MXFP4/MXFP8) would otherwise nibble-pack it and halve the
    //     2560 that ComfyUI matches on.
    //   - img_in / txt_in / proj_out: patch embed/unembed and text input proj.
    //   - norm_out.linear: final AdaLN modulation.
    //   - time_text_embed: timestep embedding MLP.
    .keys_hiprec = &.{
        "txt_norm.weight",
        "img_in.",
        "txt_in.",
        "proj_out.",
        "norm_out.linear",
        "time_text_embed",
    },
    // RMSNorm scales, as for Qwen-Image (shared block implementation).
    .upcast_from_bf16 = &.{
        "txt_norm.weight",
        ".norm_k.weight",
        ".norm_q.weight",
        ".norm_added_k.weight",
        ".norm_added_q.weight",
    },
};

pub const ernie = Arch{
    .name = "ernie",
    .keys_detect = &.{
        &.{
            "adaLN_modulation.1.weight",
            "x_embedder.proj.weight",
            "text_proj.weight",
            "layers.0.mlp.linear_fc2.weight",
        },
    },
    .shape_fix = true,
    .threshhold = null,
    .upcast_from_bf16 = &.{
        ".adaLN_sa_ln.weight",
        ".adaLN_mlp_ln.weight",
    },
};

pub const krea2 = Arch{
    .name = "krea2",
    // Detected on the native (ComfyUI single-file) naming used by Krea2 checkpoints.
    // qknorm/txtfusion are unique to Krea2, so two keys are enough to disambiguate.
    .keys_detect = &.{
        &.{
            "blocks.0.attn.qknorm.qnorm.scale",
            "txtfusion.projector.weight",
        },
    },
    .shape_fix = true,
    .threshhold = null,
    .keys_hiprec = &.{
        "txtfusion", // entire text-fusion / conditioning tower
        "tmlp", // timestep MLP (+ txtmlp text MLP)
        "tproj", // timestep projection
        "first.", // input projection (also shape-sensitive: ComfyUI reads in_channels here)
        "last.", // output projection
        ".projector",
    },
    // RMSNorm (q/k) and LayerNorm scales are precision-sensitive; keep them fp32.
    .upcast_from_bf16 = &.{
        ".qknorm.qnorm.scale",
        ".qknorm.knorm.scale",
        ".prenorm.scale",
        ".postnorm.scale",
    },
    // ComfyUI infers in_channels from first.weight.shape[1] (=64); NVFP4 nibble-packing
    // halves that dimension, so keep it as BF16 to preserve the shape.
    .keys_nvfp4_passthrough = &.{
        "first.weight",
    },
};

// MiniMax H3: a single-stream packed-token DiT that denoises video (24ch, patch
// 1x2x2) and stereo audio (32ch) latents jointly, conditioned on Qwen3-VL layer-50
// hidden states. Every weight is 1-D or 2-D — there are no convolutions — so the
// GGUF shape fix is not needed. `minimax_h3` is the `image_model` string ComfyUI
// assigns, used verbatim as `general.architecture`.
//
// The paired detect keys are the same ones ComfyUI keys off, and no other
// architecture carries an audio and a video patch projection side by side.
//
// Two checkpoint forms exist and the policy has to cover both. The full form
// stores each block's AdaLN projection at the time-embedding width (2688), which
// is 13B of the model's 33B parameters and must stay quantizable. The pruned form
// replaces it with 8 coordinates on a shared basis of the time-embedding curve
// (`adaln_t_table`), which is where keys_hiprec_narrow comes in: a 256-element
// block over 8-wide rows would mix the shift, scale and gate of 32 unrelated
// modulation rows. ComfyUI's own int8 bake of the pruned form leaves it at fp16.
pub const minimax_h3 = Arch{
    .name = "minimax_h3",
    .keys_detect = &.{
        &.{ "video_patch_proj.weight", "audio_patch_proj.weight" },
    },
    .threshhold = null,
    // Matches the set ComfyUI's reference int8_convrot bake leaves unquantized: the
    // 50 backbone blocks' qkv/out/fc1/fc2 are the target, everything else is the
    // conditioning and IO path.
    //
    // token_refiner is the one entry here that is not obviously worth its size, and
    // it is protected on the reference's authority rather than on a measurement. It
    // is two blocks structurally identical to the 50 in the backbone, ~4% of the
    // parameters, and its bf16 costs only 7% of an int8 safetensors file - but the
    // GGUF path upcasts bf16 to f32, which turns it into 22% of a q4_k file. What
    // would settle it is per-tensor damage for its eight linears against the
    // backbone's; until then it stays protected.
    .keys_hiprec = &.{
        "video_patch_proj", // fp32 in the checkpoint; ComfyUI reads shape[0] for hidden_size
        "audio_patch_proj",
        "condition_proj", // ComfyUI reads shape[1] for text_dim
        "adaln_t_table",
        "rope.inv_freq",
        "time_embedder", // absent from pruned checkpoints
        "token_refiner",
        "final_layer", // fp32 output heads plus their AdaLN
    },
    .keys_hiprec_narrow = &.{
        .{ .key = "adaln_proj.linear.weight", .below_cols = 256 },
    },
    // RMSNorm scales: per-head q/k norms and the two per-block stream norms.
    .upcast_from_bf16 = &.{
        ".attn.q_norm.weight",
        ".attn.k_norm.weight",
        ".norm1.weight",
        ".norm2.weight",
        ".final_norm.weight",
        "final_layer.norm.weight",
    },
    // NVFP4 nibble-packing halves the contiguous dimension. condition_proj.weight
    // shape[1] is text_dim and time_embedder.proj_in.weight shape[1] is
    // timestep_input_dim; both are already high-precision above, listed here so the
    // requirement survives a change to that list.
    .keys_nvfp4_passthrough = &.{
        "condition_proj.weight",
        "time_embedder.proj_in.weight",
    },
};

// SenseNova U1.5: a Mixture-of-Transformers unified model. One Qwen2-shaped
// backbone (42 layers, 4096 hidden, 12288 FFN, 32 heads over 8 KV heads) holds
// every attention and MLP weight twice - a bare copy that encodes the prompt and
// any reference images into a prefix KV cache, and a `_mot_gen` copy that
// denoises. Both branches are the model and both quantize; together they are
// 16.2B of its 17.5B parameters. It denoises pixels directly, so a checkpoint
// carries no VAE and no separate text encoder, and its keys sit at the top level
// with no `model.diffusion_model.` prefix.
//
// `sensenova_u15` is the `image_model` string ComfyUI assigns, used verbatim as
// `general.architecture`.
//
// ComfyUI additionally guards detection on two dimensions (patch_embedding rows
// 1024, q_proj_mot_gen rows 4096) because it is picking one branch of a long
// if-else. We match on names alone: nothing else here has a `_mot_gen` anything,
// and a `shape_detect` would make this arch unreachable from the name-only
// entry points. A larger MoT variant would therefore be labelled u1.5 and then
// rejected by ComfyUI's own shape guard, which is the right place to fail.
//
// The whole conditioning and pixel IO path is kept high-precision: both vision
// encoders, the timestep and noise-scale embedders, and the fm_head decoder.
// That is everything outside the backbone bar the token embedding, and it is
// 81M parameters - 0.46% of the model - so it is the same trade the mageflow
// and minimax_h3 entries make. `fm_modules.` covers the generation-side vision
// encoder, both embedders and fm_head; `vision_model.` covers the prefix-side
// encoder (the underscore in `vision_model_mot_gen` keeps the two disjoint).
//
// `language_model.lm_head.weight` is deliberately NOT ignored. ComfyUI pops it
// because its module tree has no such attribute and it never samples a token -
// the prompt template hardcodes an empty <think> block and walks straight into
// <img>. Keeping it costs ~350 MB at q4_k and leaves the checkpoint able to do
// the text half of the model. `model.norm` (as against `norm_mot_gen`) is dead
// for the same reason and kept for the same reason.
pub const sensenova_u15 = Arch{
    .name = "sensenova_u15",
    // Conv weights flatten to a contiguous extent of 2 or 3 (dense_embedding,
    // fm_head), which ggml will not accept as ne[0] for a block-quantized
    // tensor. ComfyUI-GGUF restores the logical shape from the recorded
    // orig_shape before it reads either detection dimension.
    .shape_fix = true,
    .keys_detect = &.{
        &.{
            "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
            "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
        },
    },
    .keys_hiprec = &.{
        "fm_modules.",
        "vision_model.",
    },
    .threshhold = null,
    // At q4_k this model renders a clean image with no relation to the prompt: the
    // prefix pass IS the conditioning, and `mlp.down_proj` alone destroys it by
    // layer 5 (taking just that weight dense lifts the 42-layer KV cosine from 0.45
    // to 0.988; every other weight kind moves it by under 0.04). It contracts over
    // the 12288-wide SiLU-gated hidden where the activation outliers live, which is
    // why llama.cpp's k-quant mixes upgrade `ffn_down` too.
    //
    // Base copy only: the `_mot_gen` tower sits at cosine 0.9889 on a fixed prefix
    // and taking its `down_proj` dense moves that by 0.0002, so the sensitivity is
    // the deep causal text pass rather than the layer's shape. The substring keeps
    // the two apart on its own - `mlp_mot_gen.down_proj` does not contain `mlp.`.
    .precision_floor = &.{
        .{ .key = "mlp.down_proj", .min = .bits8 },
    },
    // RMSNorm scales: each branch's per-head q/k norms (over half a head_dim,
    // split again for the h/w rope halves), its two per-block stream norms, and
    // the backbone's two final norms. Every one sits under the size threshold,
    // so a bf16 source reaches f32 without this; it bites on an f16 repack.
    .upcast_from_bf16 = &.{
        ".q_norm.weight",
        ".q_norm_mot_gen.weight",
        ".q_norm_hw.weight",
        ".q_norm_hw_mot_gen.weight",
        ".k_norm.weight",
        ".k_norm_mot_gen.weight",
        ".k_norm_hw.weight",
        ".k_norm_hw_mot_gen.weight",
        ".input_layernorm.weight",
        ".input_layernorm_mot_gen.weight",
        ".post_attention_layernorm.weight",
        ".post_attention_layernorm_mot_gen.weight",
        ".model.norm.weight",
        ".model.norm_mot_gen.weight",
    },
};

/// List of all known architectures, in detection priority order
pub const arch_list = [_]*const Arch{
    &flux,
    &sd3,
    &aura,
    &hidream,
    &anima,
    &cosmos,
    &ltx2,
    &ltxv,
    &hyvid,
    &wan,
    &sdxl,
    &sd1,
    &lumina2,
    &mageflow,
    &qwen,
    &ernie,
    &krea2,
    &minimax_h3,
    &sensenova_u15,
};

/// Core matcher: names must match, and any `shape_detect` rules must hold.
/// `tensors` is null when only names are known; an architecture that needs
/// shapes then cannot match, since we have no way to confirm its constraints.
fn archMatches(arch: *const Arch, names: []const []const u8, tensors: ?[]const types.Tensor) bool {
    if (!arch.matches(names)) return false;
    if (arch.shape_detect.len == 0) return true;
    const ts = tensors orelse return false;
    return arch.shapesMatch(ts);
}

fn detectImpl(names: []const []const u8, tensors: ?[]const types.Tensor) ?*const Arch {
    for (arch_list) |arch| {
        if (archMatches(arch, names, tensors)) return arch;
    }
    return null;
}

/// Detect architecture from a list of tensor names.
/// Returns the matching Arch or null if unknown.
///
/// Names alone cannot distinguish architectures that share a tensor-name set
/// (e.g. Mage-Flow vs Qwen-Image); those declare `shape_detect` and are only
/// reachable through `detectArchFromTensors`/`detectArchFromTensorsOrError`.
pub fn detectArch(tensor_names: []const []const u8) ?*const Arch {
    return detectImpl(tensor_names, null);
}

/// Detect architecture from a tensor list using an allocator for large models
pub fn detectArchFromTensors(tensors: []const types.Tensor, allocator: std.mem.Allocator) !?*const Arch {
    const names = try allocator.alloc([]const u8, tensors.len);
    defer allocator.free(names);

    for (tensors, 0..) |t, i| {
        names[i] = t.name;
    }
    return detectImpl(names, tensors);
}

/// Detect architecture and return error if not found or invalid
pub fn detectArchOrError(tensor_names: []const []const u8) ArchError!*const Arch {
    return detectImpl(tensor_names, null) orelse ArchError.UnknownArchitecture;
}

/// Detect architecture from tensors and return error if not found or invalid
pub fn detectArchFromTensorsOrError(tensors: []const types.Tensor, allocator: std.mem.Allocator) ArchError!*const Arch {
    const names = allocator.alloc([]const u8, tensors.len) catch return ArchError.OutOfMemory;
    defer allocator.free(names);

    for (tensors, 0..) |t, i| {
        names[i] = t.name;
    }

    return detectImpl(names, tensors) orelse ArchError.UnknownArchitecture;
}

/// Error type for architecture validation
pub const ArchError = error{
    UnknownArchitecture,
    InvalidModelFormat,
    OutOfMemory,
};

/// Fallback used when allow_unknown_arch is set and no architecture matches.
/// Has no detection keys, no ignored keys, no shape fix, and no sensitivities.
pub const generic_arch: Arch = .{
    .name = "unknown",
    .keys_detect = &.{},
    .threshhold = null,
};

/// Strip prefixes from a tensor name (e.g. "model.diffusion_model.", etc.)
pub fn stripPrefix(name: []const u8) []const u8 {
    // Prefixes for mixed state dict
    const mixed_prefixes = [_][]const u8{
        "model.diffusion_model.",
        "model.",
    };

    // Prefixes for uniform state dict (would need to check if ALL tensors have this)
    // For now, we'll just handle mixed prefixes
    const uniform_prefixes = [_][]const u8{
        "net.",
    };

    // Check mixed prefixes (any tensor can have these)
    for (mixed_prefixes) |prefix| {
        if (std.mem.startsWith(u8, name, prefix)) {
            return name[prefix.len..];
        }
    }

    // Check uniform prefixes
    for (uniform_prefixes) |prefix| {
        if (std.mem.startsWith(u8, name, prefix)) {
            return name[prefix.len..];
        }
    }

    // No prefix found, return original name
    return name;
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// Tests
// ============================================================================

test "detect flux architecture" {
    const names = [_][]const u8{"double_blocks.0.img_attn.proj.weight"};
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("flux", arch.?.name);
}

test "detect sdxl architecture" {
    const names = [_][]const u8{
        "down_blocks.0.downsamplers.0.conv.weight",
        "add_embedding.linear_1.weight",
    };
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("sdxl", arch.?.name);
    try std.testing.expect(arch.?.shape_fix);
}

test "detect qwen architecture" {
    // this will match flux as well, but has a banned key, so it should skip flux and match qwen
    const names = [_][]const u8{
        "time_text_embed.timestep_embedder.linear_2.weight",
        "transformer_blocks.0.attn.norm_added_q.weight",
        "transformer_blocks.0.img_mlp.net.0.proj.weight",
        "transformer_blocks.0.attn.norm_added_k.weight",
        "transformer_blocks.0.attn.norm_added_k.weight",
    };
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("qwen_image", arch.?.name);
    try std.testing.expect(arch.?.shape_fix);
}

test "detect architecture from tensors with allocator" {
    const allocator = std.testing.allocator;
    const tensors = [_]types.Tensor{
        .{ .name = "double_blocks.0.img_attn.proj.weight", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
    };
    const arch = try detectArchFromTensors(&tensors, allocator);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("flux", arch.?.name);
}

test "detect architecture with prefix using allocator" {
    const allocator = std.testing.allocator;
    const tensors = [_]types.Tensor{
        .{ .name = "model.diffusion_model.double_blocks.0.img_attn.proj.weight", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
    };
    const arch = try detectArchFromTensors(&tensors, allocator);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("flux", arch.?.name);
}

test "high precision key detection" {
    try std.testing.expect(hidream.isHighPrecision("some.ff_i.gate.weight"));
    try std.testing.expect(!hidream.isHighPrecision("other.key"));
}

test "ignore key detection" {
    try std.testing.expect(cosmos.shouldIgnore("layer._extra_state.data"));
    try std.testing.expect(cosmos.shouldIgnore("accum_grad"));
    try std.testing.expect(!cosmos.shouldIgnore("normal.weight"));
}

test "anima vs cosmos detection priority" {
    // A base-Cosmos state dict (no llm_adapter) must resolve to cosmos.
    const cosmos_only = [_][]const u8{
        "net.blocks.0.mlp.layer1.weight",
        "net.blocks.0.adaln_modulation_cross_attn.1.weight",
    };
    try std.testing.expectEqualStrings("cosmos", detectArch(&cosmos_only).?.name);

    // Adding the llm_adapter discriminator must flip detection to anima, even
    // though the cosmos key set is still fully present.
    const anima_sd = [_][]const u8{
        "model.diffusion_model.blocks.0.mlp.layer1.weight",
        "model.diffusion_model.blocks.0.adaln_modulation_cross_attn.1.weight",
        "model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight",
    };
    try std.testing.expectEqualStrings("anima", detectArch(&anima_sd).?.name);
}

test "anima keeps the whole llm_adapter high-precision" {
    // The entire adapter must be unquantized (Forge Neo reroutes it into the
    // text-encoder MixedPrecision path; a quantized adapter breaks its attention).
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.llm_adapter.embed.weight"));
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.llm_adapter.blocks.0.cross_attn.q_proj.weight"));
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.llm_adapter.out_proj.weight"));
    // Backbone weights stay quantizable.
    try std.testing.expect(!anima.isHighPrecision("model.diffusion_model.blocks.5.mlp.layer1.weight"));
    // Reference (silveroxides) hiprec layers: first block, final layer, embedders.
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.blocks.0.mlp.layer1.weight"));
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.final_layer.linear.weight"));
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.t_embedder.1.linear_1.weight"));
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.x_embedder.proj.1.weight"));
    // Only block 1's adaln modulation is protected, not the rest of block 1.
    try std.testing.expect(anima.isHighPrecision("model.diffusion_model.blocks.1.adaln_modulation_mlp.1.weight"));
    try std.testing.expect(!anima.isHighPrecision("model.diffusion_model.blocks.1.mlp.layer1.weight"));
}

test "banned key detection with allocator" {
    const tensors_with_banned = [_]types.Tensor{
        .{ .name = "double_blocks.0.img_attn.proj.weight", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
        .{ .name = "transformer_blocks.0.attn.norm_added_k.weight", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
    };
    try std.testing.expect(hasBannedKeysInTensors(&flux, &tensors_with_banned));

    const tensors_without_banned = [_]types.Tensor{
        .{ .name = "double_blocks.0.img_attn.proj.weight", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
        .{ .name = "some.other.tensor", .type = "F16", .dims = &.{}, .size = 0, .offset = 0 },
    };
    try std.testing.expect(! hasBannedKeysInTensors(&flux, &tensors_without_banned));
}

test "qwen upcast from bf16 - exact match" {
    try std.testing.expect(qwen.shouldUpcast("txt_norm.weight"));
    try std.testing.expect(!qwen.shouldUpcast("txt_norm.bias"));
    try std.testing.expect(!qwen.shouldUpcast("some.txt_norm.weight")); // not exact
}

test "qwen upcast from bf16 - suffix match" {
    try std.testing.expect(qwen.shouldUpcast("transformer_blocks.0.attn.norm_k.weight"));
    try std.testing.expect(qwen.shouldUpcast("transformer_blocks.5.attn.norm_q.weight"));
    try std.testing.expect(qwen.shouldUpcast("transformer_blocks.0.attn.norm_added_k.weight"));
    try std.testing.expect(qwen.shouldUpcast("transformer_blocks.0.attn.norm_added_q.weight"));
}

test "qwen upcast from bf16 - no false positives" {
    try std.testing.expect(!qwen.shouldUpcast("transformer_blocks.0.attn.norm_k.bias"));
    try std.testing.expect(!qwen.shouldUpcast("some.other.weight"));
    try std.testing.expect(!qwen.shouldUpcast("norm_k.weight.extra")); // suffix only, not contains
}

test "detect ltxv v1 architecture" {
    const names = [_][]const u8{
        "adaln_single.emb.timestep_embedder.linear_2.weight",
        "transformer_blocks.27.scale_shift_table",
        "caption_projection.linear_2.weight",
    };
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("ltxv", arch.?.name);
}

test "detect ltx2 architecture" {
    const names = [_][]const u8{
        "model.diffusion_model.adaln_single.emb.timestep_embedder.linear_2.weight",
        "model.diffusion_model.transformer_blocks.47.scale_shift_table",
        "model.diffusion_model.patchify_proj.weight",
        "model.diffusion_model.audio_adaln_single.linear.weight",
    };
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    // Both ltxv and ltx2 write "ltxv" as general.architecture for ComfyUI compatibility.
    try std.testing.expectEqualStrings("ltxv", arch.?.name);
    // But it must resolve to the ltx2 constant, not ltxv, to get the correct hiprec list.
    try std.testing.expectEqual(&ltx2, arch.?);
}

// Mage-Flow and Qwen-Image share an identical tensor-name set, so the only
// thing separating them is txt_norm.weight[0] (2560 vs 3584) and
// proj_out.weight[0] (128 vs 64) — the same pair ComfyUI keys off.
// The dim arrays live in per-instantiation static storage: `Tensor.dims` is a
// slice, so function-local arrays would dangle once the helper returns.
fn mmditTensors(comptime txt_norm_dim: usize, comptime proj_out_rows: usize) [5]types.Tensor {
    const dims = struct {
        var img_mlp = [_]usize{ 12288, 3072 };
        var linear2 = [_]usize{ 3072, 3072 };
        var norm_added_q = [_]usize{128};
        var txt_norm = [_]usize{txt_norm_dim};
        var proj_out = [_]usize{ proj_out_rows, 3072 };
    };
    return .{
        .{ .name = "time_text_embed.timestep_embedder.linear_2.weight", .type = "BF16", .dims = &dims.linear2, .size = 0, .offset = 0 },
        .{ .name = "transformer_blocks.0.attn.norm_added_q.weight", .type = "BF16", .dims = &dims.norm_added_q, .size = 0, .offset = 0 },
        .{ .name = "transformer_blocks.0.img_mlp.net.0.proj.weight", .type = "BF16", .dims = &dims.img_mlp, .size = 0, .offset = 0 },
        .{ .name = "txt_norm.weight", .type = "BF16", .dims = &dims.txt_norm, .size = 0, .offset = 0 },
        .{ .name = "proj_out.weight", .type = "BF16", .dims = &dims.proj_out, .size = 0, .offset = 0 },
    };
}

test "mage_flow vs qwen-image disambiguation is by shape only" {
    const allocator = std.testing.allocator;

    var mage = mmditTensors(2560, 128);
    try std.testing.expectEqualStrings("mage_flow", (try detectArchFromTensors(&mage, allocator)).?.name);

    // Qwen-Image: same names, different dims — must not be claimed by mage_flow.
    var qwen_image = mmditTensors(3584, 64);
    try std.testing.expectEqualStrings("qwen_image", (try detectArchFromTensors(&qwen_image, allocator)).?.name);

    // One matching dimension is not enough; both rules must hold.
    var half_match = mmditTensors(2560, 64);
    try std.testing.expectEqualStrings("qwen_image", (try detectArchFromTensors(&half_match, allocator)).?.name);
}

test "shape rules reject tensors with missing or too-few dims" {
    var no_dims = [_]usize{};
    var proj_out_dims = [_]usize{ 128, 3072 };
    const tensors = [_]types.Tensor{
        .{ .name = "txt_norm.weight", .type = "BF16", .dims = &no_dims, .size = 0, .offset = 0 },
        .{ .name = "proj_out.weight", .type = "BF16", .dims = &proj_out_dims, .size = 0, .offset = 0 },
    };
    try std.testing.expect(!mageflow.shapesMatch(&tensors));

    // Absent tensor also fails, rather than silently passing.
    try std.testing.expect(!mageflow.shapesMatch(tensors[1..]));
}

test "mage_flow keeps the conditioning and IO path high-precision" {
    const protected = [_][]const u8{
        "txt_norm.weight",
        "img_in.weight",
        "txt_in.weight",
        "proj_out.weight",
        "norm_out.linear.weight",
        "time_text_embed.timestep_embedder.linear_1.weight",
        "model.diffusion_model.proj_out.bias",
    };
    for (protected) |k| try std.testing.expect(mageflow.isHighPrecision(k));

    // The 12 double-stream blocks are the quantization target.
    const backbone = [_][]const u8{
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.11.img_mlp.net.0.proj.weight",
        "transformer_blocks.5.txt_mod.1.weight",
        "transformer_blocks.7.attn.add_v_proj.weight",
    };
    for (backbone) |k| try std.testing.expect(!mageflow.isHighPrecision(k));
}

test "mage_flow upcasts rmsnorm scales" {
    try std.testing.expect(mageflow.shouldUpcast("txt_norm.weight"));
    try std.testing.expect(mageflow.shouldUpcast("transformer_blocks.0.attn.norm_q.weight"));
    try std.testing.expect(mageflow.shouldUpcast("transformer_blocks.3.attn.norm_added_k.weight"));
    try std.testing.expect(!mageflow.shouldUpcast("transformer_blocks.0.attn.to_q.weight"));
}

test "detect krea2 architecture" {
    const names = [_][]const u8{
        "blocks.0.attn.qknorm.qnorm.scale",
        "txtfusion.projector.weight",
    };
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("krea2", arch.?.name);
    try std.testing.expect(arch.?.shape_fix);
}

test "detect krea2 architecture with prefix" {
    const names = [_][]const u8{
        "model.diffusion_model.blocks.0.attn.qknorm.qnorm.scale",
        "model.diffusion_model.txtfusion.projector.weight",
    };
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("krea2", arch.?.name);
}

test "krea2 upcast from bf16 - norm scales" {
    try std.testing.expect(krea2.shouldUpcast("blocks.0.attn.qknorm.qnorm.scale"));
    try std.testing.expect(krea2.shouldUpcast("blocks.27.attn.qknorm.knorm.scale"));
    try std.testing.expect(krea2.shouldUpcast("blocks.0.prenorm.scale"));
    try std.testing.expect(krea2.shouldUpcast("blocks.0.postnorm.scale"));
    try std.testing.expect(!krea2.shouldUpcast("blocks.0.attn.wq.weight"));
}

test "krea2 nvfp4 passthrough - first.weight" {
    try std.testing.expect(krea2.isNvfp4Passthrough("first.weight"));
    try std.testing.expect(krea2.isNvfp4Passthrough("model.diffusion_model.first.weight"));
    try std.testing.expect(!krea2.isNvfp4Passthrough("blocks.0.attn.wq.weight"));
}

test "detect minimax_h3 architecture" {
    const names = [_][]const u8{
        "video_patch_proj.weight",
        "audio_patch_proj.weight",
        "blocks.0.attn.qkv_proj.weight",
    };
    const arch = detectArch(&names);
    try std.testing.expect(arch != null);
    try std.testing.expectEqualStrings("minimax_h3", arch.?.name);
    // No convolutions, so the GGUF flat-block reshape must stay off.
    try std.testing.expect(!arch.?.shape_fix);
}

test "detect minimax_h3 architecture with prefix" {
    const names = [_][]const u8{
        "model.diffusion_model.video_patch_proj.weight",
        "model.diffusion_model.audio_patch_proj.weight",
    };
    try std.testing.expectEqualStrings("minimax_h3", detectArch(&names).?.name);
}

test "minimax_h3 keeps the conditioning and IO path high-precision" {
    const protected = [_][]const u8{
        "video_patch_proj.weight",
        "audio_patch_proj.bias",
        "condition_proj.weight",
        "adaln_t_table",
        "rope.inv_freq",
        "time_embedder.proj_in.weight",
        "token_refiner.blocks.1.mlp.fc1.weight",
        "token_refiner.final_norm.weight",
        "final_layer.video_out.weight",
        "final_layer.audio_out.weight",
        "model.diffusion_model.final_layer.adaln_proj.linear.weight",
    };
    for (protected) |k| try std.testing.expect(minimax_h3.isHighPrecision(k));

    // The 50 backbone blocks' four linears are the quantization target.
    const backbone = [_][]const u8{
        "blocks.0.attn.qkv_proj.weight",
        "blocks.49.attn.out_proj.weight",
        "blocks.13.mlp.fc1.weight",
        "blocks.7.mlp.fc2.weight",
        "model.diffusion_model.blocks.31.attn.qkv_proj.weight",
    };
    for (backbone) |k| try std.testing.expect(!minimax_h3.isHighPrecision(k));
}

test "minimax_h3 protects adaln_proj only in its pruned curve form" {
    // Pruned: 8 coordinates on the shared time-embedding basis.
    var curve = [_]usize{ 96768, 8 };
    try std.testing.expect(minimax_h3.isNarrowHighPrecision("blocks.0.adaln_proj.linear.weight", &curve));

    // Full: the time-embedding width, 13B parameters that have to stay quantizable.
    var full = [_]usize{ 96768, 2688 };
    try std.testing.expect(!minimax_h3.isNarrowHighPrecision("blocks.0.adaln_proj.linear.weight", &full));

    // The rule is scoped to that one key, and a shapeless tensor never matches.
    try std.testing.expect(!minimax_h3.isNarrowHighPrecision("blocks.0.mlp.fc1.weight", &curve));
    try std.testing.expect(!minimax_h3.isNarrowHighPrecision("blocks.0.adaln_proj.linear.weight", &.{}));

    // The final layer's AdaLN is already unconditionally protected, either way.
    try std.testing.expect(minimax_h3.isHighPrecision("final_layer.adaln_proj.linear.weight"));
}

test "minimax_h3 upcasts rmsnorm scales" {
    try std.testing.expect(minimax_h3.shouldUpcast("blocks.0.attn.q_norm.weight"));
    try std.testing.expect(minimax_h3.shouldUpcast("blocks.49.attn.k_norm.weight"));
    try std.testing.expect(minimax_h3.shouldUpcast("blocks.0.norm1.weight"));
    try std.testing.expect(minimax_h3.shouldUpcast("blocks.0.norm2.weight"));
    try std.testing.expect(minimax_h3.shouldUpcast("token_refiner.final_norm.weight"));
    try std.testing.expect(minimax_h3.shouldUpcast("final_layer.norm.weight"));
    try std.testing.expect(!minimax_h3.shouldUpcast("blocks.0.attn.qkv_proj.weight"));
}

test "minimax_h3 nvfp4 passthrough keeps the detected dimensions" {
    try std.testing.expect(minimax_h3.isNvfp4Passthrough("condition_proj.weight"));
    try std.testing.expect(minimax_h3.isNvfp4Passthrough("time_embedder.proj_in.weight"));
    try std.testing.expect(!minimax_h3.isNvfp4Passthrough("blocks.0.mlp.fc2.weight"));
}

test "krea2 high-precision policy matches ComfyUI reference (backbone-only quant)" {
    // Protected (kept high precision) — everything outside the main image DiT backbone.
    const protected = [_][]const u8{
        "txtfusion.layerwise_blocks.0.attn.wq.weight",
        "txtfusion.refiner_blocks.1.mlp.down.weight",
        "txtfusion.projector.weight",
        "tmlp.0.weight",
        "txtmlp.1.weight",
        "tproj.0.weight",
        "first.weight",
        "last.linear.weight",
        "model.diffusion_model.txtfusion.refiner_blocks.0.attn.wo.weight",
    };
    for (protected) |k| try std.testing.expect(krea2.isHighPrecision(k));

    // Quantized — the image DiT backbone linears must NOT be protected.
    const backbone = [_][]const u8{
        "blocks.0.attn.wq.weight",
        "blocks.27.attn.wo.weight",
        "blocks.13.mlp.up.weight",
        "blocks.5.mlp.down.weight",
        "blocks.0.attn.gate.weight",
        "model.diffusion_model.blocks.9.mlp.gate.weight",
    };
    for (backbone) |k| try std.testing.expect(!krea2.isHighPrecision(k));
}
test "sensenova_u15 detection needs both MoT discriminators" {
    const full = [_][]const u8{
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
        "fm_modules.fm_head.conv1.weight",
        "language_model.model.embed_tokens.weight",
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
        "language_model.model.layers.0.mlp.gate_proj.weight",
        "vision_model.embeddings.patch_embedding.weight",
    };
    try std.testing.expectEqualStrings("sensenova_u15", detectArch(&full).?.name);

    // The vision encoder alone is not enough: a plain VLM has one too.
    const no_gen_branch = [_][]const u8{
        "fm_modules.vision_model_mot_gen.embeddings.patch_embedding.weight",
        "language_model.model.layers.0.self_attn.q_proj.weight",
    };
    try std.testing.expect(detectArch(&no_gen_branch) == null);
}

test "sensenova_u15 protects the IO path and quantizes both MoT branches" {
    const protected = [_][]const u8{
        "vision_model.embeddings.patch_embedding.weight",
        "vision_model.embeddings.dense_embedding.weight",
        "fm_modules.vision_model_mot_gen.embeddings.dense_embedding.weight",
        "fm_modules.timestep_embedder.mlp.0.weight",
        "fm_modules.noise_scale_embedder.mlp.2.weight",
        "fm_modules.fm_head.conv1.weight",
        "fm_modules.fm_head.conv2.weight",
    };
    for (protected) |k| try std.testing.expect(sensenova_u15.isHighPrecision(k));

    // Both copies of every backbone weight are the model, and both quantize.
    const backbone = [_][]const u8{
        "language_model.model.layers.0.self_attn.q_proj.weight",
        "language_model.model.layers.0.self_attn.q_proj_mot_gen.weight",
        "language_model.model.layers.41.self_attn.o_proj_mot_gen.weight",
        "language_model.model.layers.20.mlp.down_proj.weight",
        "language_model.model.layers.20.mlp_mot_gen.down_proj.weight",
        "language_model.lm_head.weight",
    };
    for (backbone) |k| try std.testing.expect(!sensenova_u15.isHighPrecision(k));
}

test "sensenova_u15 upcasts every RMSNorm scale of both branches" {
    const scales = [_][]const u8{
        "language_model.model.layers.0.self_attn.q_norm.weight",
        "language_model.model.layers.0.self_attn.q_norm_mot_gen.weight",
        "language_model.model.layers.0.self_attn.q_norm_hw.weight",
        "language_model.model.layers.0.self_attn.q_norm_hw_mot_gen.weight",
        "language_model.model.layers.7.self_attn.k_norm.weight",
        "language_model.model.layers.7.self_attn.k_norm_mot_gen.weight",
        "language_model.model.layers.7.self_attn.k_norm_hw.weight",
        "language_model.model.layers.7.self_attn.k_norm_hw_mot_gen.weight",
        "language_model.model.layers.3.input_layernorm.weight",
        "language_model.model.layers.3.input_layernorm_mot_gen.weight",
        "language_model.model.layers.3.post_attention_layernorm.weight",
        "language_model.model.layers.3.post_attention_layernorm_mot_gen.weight",
        "language_model.model.norm.weight",
        "language_model.model.norm_mot_gen.weight",
    };
    for (scales) |k| try std.testing.expect(sensenova_u15.shouldUpcast(k));

    try std.testing.expect(!sensenova_u15.shouldUpcast("language_model.model.layers.3.mlp.up_proj.weight"));
    try std.testing.expect(!sensenova_u15.shouldUpcast("language_model.model.embed_tokens.weight"));
}
