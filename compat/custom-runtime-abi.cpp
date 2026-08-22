#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <dlfcn.h>

// Compatibility exports for pairing the full DFlash2 server frontend with the
// stripped advanced-quantizer llama runtime. Keep this bridge deliberately
// small: it covers only APIs referenced by the frontend that are absent from
// the custom runtime's otherwise compatible ABI.

namespace {

struct advanced_model_params {
    ggml_backend_dev_t * devices;
    const llama_model_tensor_buft_override * tensor_buft_overrides;
    int32_t n_gpu_layers;
    llama_split_mode split_mode;
    int32_t main_gpu;
    const float * tensor_split;
    llama_progress_callback progress_callback;
    void * progress_callback_user_data;
    const llama_model_kv_override * kv_overrides;
    bool vocab_only;
    bool use_mmap;
    bool use_mmap_prefetch;
    bool use_direct_io;
    bool use_mlock;
    bool check_tensors;
    bool use_extra_bufts;
    bool no_host;
    bool no_alloc;
};

struct advanced_context_params {
    uint32_t n_ctx;
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    uint32_t n_rs_seq;
    uint32_t n_outputs_max;
    int32_t n_threads;
    int32_t n_threads_batch;
    llama_context_type ctx_type;
    llama_rope_scaling_type rope_scaling_type;
    enum llama_pooling_type pooling_type;
    llama_attention_type attention_type;
    llama_flash_attn_type flash_attn_type;
    float rope_freq_base;
    float rope_freq_scale;
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;
    uint32_t yarn_orig_ctx;
    float defrag_thold;
    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
    ggml_type type_k;
    ggml_type type_v;
    ggml_abort_callback abort_callback;
    void * abort_callback_data;
    bool embeddings;
    bool offload_kqv;
    bool no_perf;
    bool op_offload;
    bool swa_full;
    bool kv_unified;
    bool graph_reuse_disable;
    llama_sampler_seq_config * samplers;
    size_t n_samplers;
    llama_context * ctx_other;
};

template<typename T>
T next_symbol(const char * name) {
    return reinterpret_cast<T>(dlsym(RTLD_NEXT, name));
}

advanced_model_params to_advanced(llama_model_params p) {
    const bool mmap = p.load_mode == LLAMA_LOAD_MODE_AUTO ||
                      p.load_mode == LLAMA_LOAD_MODE_MMAP ||
                      p.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK;
    return {
        p.devices,
        p.tensor_buft_overrides,
        p.n_gpu_layers,
        p.split_mode,
        p.main_gpu,
        p.tensor_split,
        p.progress_callback,
        p.progress_callback_user_data,
        p.kv_overrides,
        p.vocab_only,
        mmap,
        mmap,
        p.load_mode == LLAMA_LOAD_MODE_DIRECT_IO,
        p.load_mode == LLAMA_LOAD_MODE_MLOCK || p.load_mode == LLAMA_LOAD_MODE_MMAP_MLOCK,
        p.check_tensors,
        p.use_extra_bufts,
        p.no_host,
        p.no_alloc,
    };
}

advanced_context_params to_advanced(llama_context_params p) {
    return {
        p.n_ctx,
        p.n_batch,
        p.n_ubatch,
        p.n_seq_max,
        p.n_rs_seq,
        p.n_outputs_max,
        p.n_threads,
        p.n_threads_batch,
        p.ctx_type,
        p.rope_scaling_type,
        p.pooling_type,
        p.attention_type,
        p.flash_attn_type,
        p.rope_freq_base,
        p.rope_freq_scale,
        p.yarn_ext_factor,
        p.yarn_attn_factor,
        p.yarn_beta_fast,
        p.yarn_beta_slow,
        p.yarn_orig_ctx,
        p.defrag_thold,
        p.cb_eval,
        p.cb_eval_user_data,
        p.type_k,
        p.type_v,
        p.abort_callback,
        p.abort_callback_data,
        p.embeddings,
        p.offload_kqv,
        p.no_perf,
        p.op_offload,
        p.swa_full,
        p.kv_unified,
        false,
        p.samplers,
        p.n_samplers,
        p.ctx_other,
    };
}

} // namespace

extern "C" llama_model_params llama_model_default_params(void) {
    using fn_t = advanced_model_params (*)();
    const advanced_model_params p = next_symbol<fn_t>("llama_model_default_params")();
    return {
        p.devices,
        p.tensor_buft_overrides,
        p.n_gpu_layers,
        p.split_mode,
        p.use_direct_io ? LLAMA_LOAD_MODE_DIRECT_IO :
            (p.use_mmap && p.use_mlock ? LLAMA_LOAD_MODE_MMAP_MLOCK :
             p.use_mmap ? LLAMA_LOAD_MODE_MMAP :
             p.use_mlock ? LLAMA_LOAD_MODE_MLOCK : LLAMA_LOAD_MODE_NONE),
        p.main_gpu,
        p.tensor_split,
        p.progress_callback,
        p.progress_callback_user_data,
        p.kv_overrides,
        p.vocab_only,
        p.check_tensors,
        p.use_extra_bufts,
        p.no_host,
        p.no_alloc,
        false,
    };
}

extern "C" llama_context_params llama_context_default_params(void) {
    using fn_t = advanced_context_params (*)();
    const advanced_context_params p = next_symbol<fn_t>("llama_context_default_params")();
    return {
        p.n_ctx,
        p.n_batch,
        p.n_ubatch,
        p.n_seq_max,
        p.n_rs_seq,
        p.n_outputs_max,
        1,
        p.n_threads,
        p.n_threads_batch,
        p.ctx_type,
        p.rope_scaling_type,
        p.pooling_type,
        p.attention_type,
        p.flash_attn_type,
        p.rope_freq_base,
        p.rope_freq_scale,
        p.yarn_ext_factor,
        p.yarn_attn_factor,
        p.yarn_beta_fast,
        p.yarn_beta_slow,
        p.yarn_orig_ctx,
        p.defrag_thold,
        p.cb_eval,
        p.cb_eval_user_data,
        p.type_k,
        p.type_v,
        p.abort_callback,
        p.abort_callback_data,
        p.embeddings,
        p.offload_kqv,
        p.no_perf,
        p.op_offload,
        p.swa_full,
        p.kv_unified,
        p.samplers,
        p.n_samplers,
        p.ctx_other,
    };
}

extern "C" llama_model * llama_model_load_from_file(
        const char * path_model,
        llama_model_params params) {
    using fn_t = llama_model * (*)(const char *, advanced_model_params);
    return next_symbol<fn_t>("llama_model_load_from_file")(path_model, to_advanced(params));
}

extern "C" llama_model * llama_model_load_from_file_ptr(
        FILE * file,
        llama_model_params params) {
    using fn_t = llama_model * (*)(FILE *, advanced_model_params);
    return next_symbol<fn_t>("llama_model_load_from_file_ptr")(file, to_advanced(params));
}

extern "C" llama_model * llama_model_load_from_splits(
        const char ** paths,
        size_t n_paths,
        llama_model_params params) {
    using fn_t = llama_model * (*)(const char **, size_t, advanced_model_params);
    return next_symbol<fn_t>("llama_model_load_from_splits")(paths, n_paths, to_advanced(params));
}

extern "C" llama_context * llama_init_from_model(
        llama_model * model,
        llama_context_params params) {
    using fn_t = llama_context * (*)(llama_model *, advanced_context_params);
    return next_symbol<fn_t>("llama_init_from_model")(model, to_advanced(params));
}

extern "C" const char * llama_version(void) {
    return "advanced-gguf-quantizer+dflash2";
}

extern "C" enum llama_ftype llama_model_ftype(const struct llama_model * /* model */) {
    // Used by the server only for its informational model-type string. Mixed
    // custom tensors do not have a single representative upstream ftype.
    return LLAMA_FTYPE_GUESSED;
}

extern "C" const char * llama_ftype_name(enum llama_ftype /* ftype */) {
    return "NVFP4/MXFP6/MXFP8 mixed";
}

extern "C" const llama_token * llama_vocab_get_suppress_tokens(
        const struct llama_vocab * /* vocab */,
        int32_t * n_suppress_tokens) {
    // The custom runtime predates model-provided suppression lists. Its vocab
    // therefore has no corresponding state to expose.
    if (n_suppress_tokens) {
        *n_suppress_tokens = 0;
    }
    return nullptr;
}

extern "C" void llama_sampler_copy(
        const struct llama_sampler * src,
        struct llama_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    // All samplers used by common_sampler_copy are cloned from the same source
    // configuration, so their interfaces match. Clone the current state, free
    // the destination state, and transplant the cloned context while retaining
    // the destination object's stable address.
    if (src->iface != dst->iface) {
        return;
    }

    llama_sampler * tmp = llama_sampler_clone(src);
    if (!tmp) {
        return;
    }

    if (dst->iface->free) {
        dst->iface->free(dst);
    }

    dst->ctx = tmp->ctx;
    tmp->ctx = nullptr;
    delete tmp;
}
