using System;
using System.Runtime.InteropServices;

namespace LlamaCSharp
{
    // Opaque pointer types
    using llama_model = IntPtr;
    using llama_context = IntPtr;
    using llama_sampler = IntPtr;
    using llama_memory_t = IntPtr;
    using llama_vocab = IntPtr;
    using llama_adapter_lora = IntPtr;
    using ggml_backend_dev_t = IntPtr;
    using ggml_backend_buffer_type_t = IntPtr;
    using ggml_threadpool_t = IntPtr;

    // Token types
    using llama_token = Int32;
    using llama_pos = Int32;
    using llama_seq_id = Int32;

   

    #region Structs

    [StructLayout(LayoutKind.Sequential)]
    public struct llama_token_data
    {
        public llama_token id;
        public float logit;
        public float p;
    }

    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct llama_token_data_array
    {
        public llama_token_data* data;
        public UIntPtr size;
        public long selected;
        [MarshalAs(UnmanagedType.I1)]
        public bool sorted;
    }

    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct llama_batch
    {
        public int n_tokens;
        public llama_token* token;
        public float* embd;
        public llama_pos* pos;
        public int* n_seq_id;
        public llama_seq_id** seq_id;
        public sbyte* logits;
    }

    [StructLayout(LayoutKind.Sequential)]
    public unsafe struct llama_model_params
    {
        public ggml_backend_dev_t* devices;
        public IntPtr tensor_buft_overrides;
        public int n_gpu_layers;
        public llama_split_mode split_mode;
        public int main_gpu;
        public float* tensor_split;
        public IntPtr progress_callback;
        public IntPtr progress_callback_user_data;
        public IntPtr kv_overrides;
        [MarshalAs(UnmanagedType.I1)]
        public bool vocab_only;
        [MarshalAs(UnmanagedType.I1)]
        public bool use_mmap;
        [MarshalAs(UnmanagedType.I1)]
        public bool use_direct_io;
        [MarshalAs(UnmanagedType.I1)]
        public bool use_mlock;
        [MarshalAs(UnmanagedType.I1)]
        public bool check_tensors;
        [MarshalAs(UnmanagedType.I1)]
        public bool use_extra_bufts;
        [MarshalAs(UnmanagedType.I1)]
        public bool no_host;
        [MarshalAs(UnmanagedType.I1)]
        public bool no_alloc;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct llama_context_params
    {
        public uint n_ctx;
        public uint n_batch;
        public uint n_ubatch;
        public uint n_seq_max;
        public int n_threads;
        public int n_threads_batch;
        public int rope_scaling_type;
        public int pooling_type;
        public int attention_type;
        public int flash_attn_type;
        public float rope_freq_base;
        public float rope_freq_scale;
        public float yarn_ext_factor;
        public float yarn_attn_factor;
        public float yarn_beta_fast;
        public float yarn_beta_slow;
        public uint yarn_orig_ctx;
        public float defrag_thold;
        public IntPtr cb_eval;
        public IntPtr cb_eval_user_data;
        public int type_k;
        public int type_v;
        public IntPtr abort_callback;
        public IntPtr abort_callback_data;
        [MarshalAs(UnmanagedType.I1)]
        public bool embeddings;
        [MarshalAs(UnmanagedType.I1)]
        public bool offload_kqv;
        [MarshalAs(UnmanagedType.I1)]
        public bool no_perf;
        [MarshalAs(UnmanagedType.I1)]
        public bool op_offload;
        [MarshalAs(UnmanagedType.I1)]
        public bool swa_full;
        [MarshalAs(UnmanagedType.I1)]
        public bool kv_unified;
        public IntPtr samplers;
        public UIntPtr n_samplers;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct llama_sampler_chain_params
    {
        [MarshalAs(UnmanagedType.I1)]
        public bool no_perf;
    }

    [StructLayout(LayoutKind.Sequential)]
    public struct llama_chat_message
    {
        public IntPtr role;
        public IntPtr content;
    }

    #endregion

    public static class Llama
    {
        private const string LibraryName = @"lib\llama.dll";

        #region Backend / Initialization

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_model_params llama_model_default_params();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_context_params llama_context_default_params();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_sampler_chain_params llama_sampler_chain_default_params();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_backend_init();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_backend_free();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern long llama_time_us();

        #endregion

        #region Model Loading / Freeing

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_model llama_model_load_from_file(
            [MarshalAs(UnmanagedType.LPUTF8Str)] string path_model,
            llama_model_params parameters);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_model_free(llama_model model);

        #endregion

        #region Context

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_context llama_init_from_model(
            llama_model model,
            llama_context_params parameters);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_free(llama_context ctx);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint llama_n_ctx(llama_context ctx);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern uint llama_n_batch(llama_context ctx);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_model llama_get_model(llama_context ctx);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_memory_t llama_get_memory(llama_context ctx);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_vocab llama_model_get_vocab(llama_model model);

        #endregion

        #region Model Info

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_model_n_ctx_train(llama_model model);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_model_n_embd(llama_model model);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_model_n_layer(llama_model model);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong llama_model_size(llama_model model);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern ulong llama_model_n_params(llama_model model);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_model_desc(llama_model model, IntPtr buf, UIntPtr buf_size);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_model_has_encoder(llama_model model);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_model_has_decoder(llama_model model);

        #endregion

        #region Vocab

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_vocab_type llama_vocab_type(llama_vocab vocab);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_vocab_n_tokens(llama_vocab vocab);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_vocab_get_text(llama_vocab vocab, llama_token token);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern float llama_vocab_get_score(llama_vocab vocab, llama_token token);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_vocab_is_eog(llama_vocab vocab, llama_token token);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_vocab_is_control(llama_vocab vocab, llama_token token);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_token llama_vocab_bos(llama_vocab vocab);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_token llama_vocab_eos(llama_vocab vocab);

        #endregion

        #region Tokenization

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int llama_tokenize(
            llama_vocab vocab,
            [MarshalAs(UnmanagedType.LPUTF8Str)] string text,
            int text_len,
            llama_token* tokens,
            int n_tokens_max,
            [MarshalAs(UnmanagedType.I1)] bool add_special,
            [MarshalAs(UnmanagedType.I1)] bool parse_special);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_token_to_piece(
            llama_vocab vocab,
            llama_token token,
            IntPtr buf,
            int length,
            int lstrip,
            [MarshalAs(UnmanagedType.I1)] bool special);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe int llama_detokenize(
            llama_vocab vocab,
            llama_token* tokens,
            int n_tokens,
            IntPtr text,
            int text_len_max,
            [MarshalAs(UnmanagedType.I1)] bool remove_special,
            [MarshalAs(UnmanagedType.I1)] bool unparse_special);

        #endregion

        #region Batch

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe llama_batch llama_batch_get_one(
            llama_token* tokens,
            int n_tokens);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_batch llama_batch_init(
            int n_tokens,
            int embd,
            int n_seq_max);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_batch_free(llama_batch batch);

        #endregion

        #region Decoding

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_encode(
            llama_context ctx,
            llama_batch batch);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern int llama_decode(
            llama_context ctx,
            llama_batch batch);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_set_n_threads(
            llama_context ctx,
            int n_threads,
            int n_threads_batch);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_synchronize(llama_context ctx);

        #endregion

        #region Logits / Embeddings

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe float* llama_get_logits(llama_context ctx);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe float* llama_get_logits_ith(llama_context ctx, int i);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe float* llama_get_embeddings(llama_context ctx);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe float* llama_get_embeddings_ith(llama_context ctx, int i);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern unsafe float* llama_get_embeddings_seq(llama_context ctx, llama_seq_id seq_id);

        #endregion

        #region Memory Management

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_memory_clear(
            llama_memory_t mem,
            [MarshalAs(UnmanagedType.I1)] bool data);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_memory_seq_rm(
            llama_memory_t mem,
            llama_seq_id seq_id,
            llama_pos p0,
            llama_pos p1);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_memory_seq_cp(
            llama_memory_t mem,
            llama_seq_id seq_id_src,
            llama_seq_id seq_id_dst,
            llama_pos p0,
            llama_pos p1);

        #endregion

        #region Samplers

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_sampler llama_sampler_chain_init(llama_sampler_chain_params parameters);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sampler_chain_add(llama_sampler chain, llama_sampler smpl);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_sampler_free(llama_sampler smpl);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_sampler llama_sampler_init_greedy();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_sampler llama_sampler_init_dist(uint seed);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_sampler llama_sampler_init_top_k(int k);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_sampler llama_sampler_init_top_p(float p, UIntPtr min_keep);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_sampler llama_sampler_init_min_p(float p, UIntPtr min_keep);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_sampler llama_sampler_init_temp(float t);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern llama_token llama_sampler_sample(
            llama_sampler smpl,
            llama_context ctx,
            int idx);

        #endregion

        #region Utilities

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern IntPtr llama_print_system_info();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_supports_mmap();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_supports_mlock();

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        [return: MarshalAs(UnmanagedType.I1)]
        public static extern bool llama_supports_gpu_offload();

        #endregion

        //This is for telling LLama to log so much

        [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
        public delegate void LlamaLogCallback(int level, IntPtr text, IntPtr user_data);

        [DllImport(LibraryName, CallingConvention = CallingConvention.Cdecl)]
        public static extern void llama_log_set(LlamaLogCallback callback, IntPtr user_data);

    }
}