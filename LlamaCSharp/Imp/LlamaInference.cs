using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;



namespace LlamaCSharp.Imp
{


    /// <summary>
    /// High-level wrapper for llama.cpp inference.
    /// </summary>
    public class LlamaInference : IDisposable
    {
        private IntPtr _model;
        private IntPtr _context;
        private IntPtr _vocab;
        private bool _disposed;

        public int ContextSize { get; private set; }
        public int VocabSize { get; private set; }
        public string ModelPath { get; private set; }

        /// <summary>
        /// Initialize Conductor with a model file.
        /// </summary>
        public LlamaInference(string modelPath, LlamaConfig config = null, int warninglevel = -1)
        {
            config ??= new LlamaConfig();
            if (config.SilentMode)
            {
                Llama.llama_log_set((level, text, user_data) =>
                {
                    // Do nothing = silent
                    // OR filter by level if you want warnings/errors only
                    if (level <= warninglevel) // Only errors (0-2 are error levels)
                    {
                        Console.Error.WriteLine(Marshal.PtrToStringUTF8(text));
                    }
                }, IntPtr.Zero);
            }
            
            ModelPath = modelPath;

            // Initialize backend once
            Llama.llama_backend_init();

            // Load model
            var modelParams = Llama.llama_model_default_params();
            modelParams.n_gpu_layers = config.GpuLayers;
            modelParams.use_mmap = config.UseMmap;
            modelParams.use_mlock = config.UseMlock;

            _model = Llama.llama_model_load_from_file(modelPath, modelParams);
            if (_model == IntPtr.Zero)
                throw new Exception($"Failed to load model from: {modelPath}");

            // Create context
            var ctxParams = Llama.llama_context_default_params();
            ctxParams.n_ctx = (uint)config.ContextSize;
            ctxParams.n_batch = (uint)config.BatchSize;
            ctxParams.n_threads = config.Threads;
            ctxParams.n_threads_batch = config.ThreadsBatch;

            _context = Llama.llama_init_from_model(_model, ctxParams);
            if (_context == IntPtr.Zero)
                throw new Exception("Failed to create context");

            _vocab = Llama.llama_model_get_vocab(_model);
            ContextSize = (int)Llama.llama_n_ctx(_context);
            VocabSize = Llama.llama_vocab_n_tokens(_vocab);
        }

        public LlamaInference(IntPtr _m, LlamaConfig config)
        {
            _model = _m;
            if (_model == IntPtr.Zero)
                throw new Exception($"Failed to load model from");

            // Create context
            var ctxParams = Llama.llama_context_default_params();
            ctxParams.n_ctx = (uint)config.ContextSize;
            ctxParams.n_batch = (uint)config.BatchSize;
            ctxParams.n_ubatch = (uint)config.UBatchSize;
            ctxParams.n_threads = config.Threads;
            ctxParams.n_threads_batch = config.ThreadsBatch;

            _context = Llama.llama_init_from_model(_model, ctxParams);
            if (_context == IntPtr.Zero)
                throw new Exception("Failed to create context");

            _vocab = Llama.llama_model_get_vocab(_model);
            ContextSize = (int)Llama.llama_n_ctx(_context);
            VocabSize = Llama.llama_vocab_n_tokens(_vocab);
        }

        /// <summary>
        /// Tokenize text into tokens.
        /// </summary>
        public unsafe int[] Tokenize(string text, bool addBos = true, bool parseSpecial = false)
        {
            // Estimate max tokens (conservative)
            int maxTokens = text.Length + (addBos ? 1 : 0);
            var tokens = new int[maxTokens];

            fixed (int* tokensPtr = tokens)
            {
                int tokenCount = Llama.llama_tokenize(
                    _vocab,
                    text,
                    text.Length,
                    tokensPtr,
                    maxTokens,
                    addBos,
                    parseSpecial
                );

                if (tokenCount < 0)
                {
                    // Need more space
                    maxTokens = -tokenCount;
                    tokens = new int[maxTokens];
                    fixed (int* tokensPtr2 = tokens)
                    {
                        tokenCount = Llama.llama_tokenize(
                            _vocab,
                            text,
                            text.Length,
                            tokensPtr2,
                            maxTokens,
                            addBos,
                            parseSpecial
                        );
                    }
                }

                Array.Resize(ref tokens, tokenCount);
                return tokens;
            }
        }

        /// <summary>
        /// Convert tokens back to text.
        /// </summary>
        public unsafe string Detokenize(int[] tokens, bool removeSpecial = false)
        {
            int bufferSize = tokens.Length * 16; // Conservative estimate
            IntPtr buffer = Marshal.AllocHGlobal(bufferSize);

            try
            {
                fixed (int* tokensPtr = tokens)
                {
                    int length = Llama.llama_detokenize(
                        _vocab,
                        tokensPtr,
                        tokens.Length,
                        buffer,
                        bufferSize,
                        removeSpecial,
                        false
                    );

                    if (length < 0)
                    {
                        // Need more space
                        bufferSize = -length;
                        Marshal.FreeHGlobal(buffer);
                        buffer = Marshal.AllocHGlobal(bufferSize);

                        length = Llama.llama_detokenize(
                            _vocab,
                            tokensPtr,
                            tokens.Length,
                            buffer,
                            bufferSize,
                            removeSpecial,
                            false
                        );
                    }

                    return Marshal.PtrToStringUTF8(buffer, length);
                }
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        /// <summary>
        /// Generate text from a prompt.
        /// </summary>
        public string Generate(string prompt, GenerationConfig config = null)
        {
            config ??= new GenerationConfig();

            var promptTokens = Tokenize(prompt, addBos: true);
            var generatedTokens = new List<int>();

            var memory = Llama.llama_get_memory(_context);
            Llama.llama_memory_clear(memory, data: false);

            var samplerParams = Llama.llama_sampler_chain_default_params();
            var sampler = Llama.llama_sampler_chain_init(samplerParams);

            try
            {
                // Add samplers in order

                if (config.RepeatPenaltyTokens > 0 && config.RepeatPenalty != 1.0f)
                {
                    Llama.llama_sampler_chain_add(
                        sampler,
                        Llama.llama_sampler_init_penalties(
                            penalty_last_n: config.RepeatPenaltyTokens,
                            penalty_repeat: config.RepeatPenalty,
                            penalty_freq: config.FrequencyPenalty,
                            penalty_present: config.PresencePenalty
                        )
                    );
                }

                
                if (config.TopK > 0)
                    Llama.llama_sampler_chain_add(sampler, Llama.llama_sampler_init_top_k(config.TopK));

              
                if (config.TopP < 1.0f)
                    Llama.llama_sampler_chain_add(sampler, Llama.llama_sampler_init_top_p(config.TopP, (UIntPtr)1));

                
                if (config.MinP > 0.0f)
                    Llama.llama_sampler_chain_add(sampler, Llama.llama_sampler_init_min_p(config.MinP, (UIntPtr)1));

                
                Llama.llama_sampler_chain_add(sampler, Llama.llama_sampler_init_temp(config.Temperature));

                
                Llama.llama_sampler_chain_add(sampler, Llama.llama_sampler_init_dist(config.Seed));

                // Evaluate prompt
                if (!EvaluateBatch(promptTokens))
                    throw new Exception("Failed to evaluate prompt");

                int eosToken = Llama.llama_vocab_eos(_vocab);

                // Generate tokens
                for (int i = 0; i < config.MaxTokens; i++)
                {
                    int nextToken = Llama.llama_sampler_sample(sampler, _context, -1);

                    if (nextToken == eosToken)
                        break;

                    generatedTokens.Add(nextToken);

                    // Check stop strings
                    if (config.StopStrings != null && config.StopStrings.Length > 0)
                    {
                        var currentText = Detokenize(generatedTokens.ToArray());

                        foreach (var stopStr in config.StopStrings)
                        {
                            if (currentText.EndsWith(stopStr))
                            {
                                int stopIndex = currentText.LastIndexOf(stopStr);
                                var truncated = currentText.Substring(0, stopIndex);
                                var cleanTokens = Tokenize(truncated, addBos: false);
                                return Detokenize(cleanTokens, removeSpecial: true);
                            }
                        }
                    }

                    if (!EvaluateBatch(new[] { nextToken }))
                        break;

                    config.OnTokenGenerated?.Invoke(nextToken);
                }

                return Detokenize(generatedTokens.ToArray(), removeSpecial: true);
            }
            finally
            {
                Llama.llama_sampler_free(sampler);
            }
        }
        /// <summary>
        /// Generate text from a prompt.
        /// </summary>
        public async Task<string> GenerateAsync( string prompt, GenerationConfig config = null, CancellationToken cancellationToken = default)
        {
            
            config ??= new GenerationConfig();

            var promptTokens = Tokenize(prompt, addBos: true);
            var generatedTokens = new List<int>();

            var memory = Llama.llama_get_memory(_context);
            Llama.llama_memory_clear(memory, data: false);

            var samplerParams = Llama.llama_sampler_chain_default_params();
            var sampler = Llama.llama_sampler_chain_init(samplerParams);

            try
            {
                // Add samplers in order

                if (config.RepeatPenaltyTokens > 0 && config.RepeatPenalty != 1.0f)
                {
                    Llama.llama_sampler_chain_add(
                        sampler,
                        Llama.llama_sampler_init_penalties(
                            penalty_last_n: config.RepeatPenaltyTokens,
                            penalty_repeat: config.RepeatPenalty,
                            penalty_freq: config.FrequencyPenalty,
                            penalty_present: config.PresencePenalty
                        )
                    );
                }

                
                if (config.TopK > 0)
                    Llama.llama_sampler_chain_add(sampler, Llama.llama_sampler_init_top_k(config.TopK));

              
                if (config.TopP < 1.0f)
                    Llama.llama_sampler_chain_add(sampler, Llama.llama_sampler_init_top_p(config.TopP, (UIntPtr)1));

                
                if (config.MinP > 0.0f)
                    Llama.llama_sampler_chain_add(sampler, Llama.llama_sampler_init_min_p(config.MinP, (UIntPtr)1));

                
                Llama.llama_sampler_chain_add(sampler, Llama.llama_sampler_init_temp(config.Temperature));

                
                Llama.llama_sampler_chain_add(sampler, Llama.llama_sampler_init_dist(config.Seed));

                // Evaluate prompt
                if (!EvaluateBatch(promptTokens))
                    throw new Exception("Failed to evaluate prompt");

                int eosToken = Llama.llama_vocab_eos(_vocab);

                // Generate tokens
                for (int i = 0; i < config.MaxTokens; i++)
                {

                    cancellationToken.ThrowIfCancellationRequested();

                    

                    int nextToken = Llama.llama_sampler_sample(sampler, _context, -1);

                    if (nextToken == eosToken)
                        break;

                    generatedTokens.Add(nextToken);

                    // Check stop strings
                    if (config.StopStrings != null && config.StopStrings.Length > 0)
                    {
                        var currentText = Detokenize(generatedTokens.ToArray());

                        foreach (var stopStr in config.StopStrings)
                        {
                            if (currentText.EndsWith(stopStr))
                            {
                                int stopIndex = currentText.LastIndexOf(stopStr);
                                var truncated = currentText.Substring(0, stopIndex);
                                var cleanTokens = Tokenize(truncated, addBos: false);
                                return Detokenize(cleanTokens, removeSpecial: true);
                            }
                        }
                    }

                    if (!EvaluateBatch(new[] { nextToken }))
                        break;

                    config.OnTokenGenerated?.Invoke(nextToken);
                    await Task.Delay(300);
                }

                return Detokenize(generatedTokens.ToArray(), removeSpecial: true);
            }
            finally
            {
                Llama.llama_sampler_free(sampler);
            }
        }

        /// <summary>
        /// Format a conversation using the model's chat template
        /// </summary>
        public unsafe string FormatChat(List<ChatMessage> messages, bool addAssistant = true)
        {
            // Convert to llama_chat_message structs
            var nativeMessages = new llama_chat_message[messages.Count];
            var handles = new List<GCHandle>();

            try
            {
                for (int i = 0; i < messages.Count; i++)
                {
                    var roleHandle = GCHandle.Alloc(
                        Encoding.UTF8.GetBytes(messages[i].Role + "\0"),
                        GCHandleType.Pinned);
                    var contentHandle = GCHandle.Alloc(
                        Encoding.UTF8.GetBytes(messages[i].Content + "\0"),
                        GCHandleType.Pinned);

                    handles.Add(roleHandle);
                    handles.Add(contentHandle);

                    nativeMessages[i].role = roleHandle.AddrOfPinnedObject();
                    nativeMessages[i].content = contentHandle.AddrOfPinnedObject();
                }

                // Allocate buffer for result
                int bufferSize = 4096;
                IntPtr buffer = Marshal.AllocHGlobal(bufferSize);

                try
                {
                    fixed (llama_chat_message* msgPtr = nativeMessages)
                    {
                        int length = Llama.llama_chat_apply_template(
                            IntPtr.Zero,  // Use model's default template
                            msgPtr,
                            (UIntPtr)messages.Count,
                            addAssistant,
                            buffer,
                            bufferSize
                        );

                        if (length < 0)
                        {
                            // Need bigger buffer
                            bufferSize = -length;
                            Marshal.FreeHGlobal(buffer);
                            buffer = Marshal.AllocHGlobal(bufferSize);

                            length = Llama.llama_chat_apply_template(
                                IntPtr.Zero,
                                msgPtr,
                                (UIntPtr)messages.Count,
                                addAssistant,
                                buffer,
                                bufferSize
                            );
                        }

                        return Marshal.PtrToStringUTF8(buffer, length);
                    }
                }
                finally
                {
                    Marshal.FreeHGlobal(buffer);
                }
            }
            finally
            {
                // Free all GC handles
                foreach (var handle in handles)
                {
                    handle.Free();
                }
            }
        }

        /// <summary>
        /// Get the model's built-in chat template (if it has one)
        /// </summary>
        public string GetChatTemplate()
        {
            var templatePtr = Llama.llama_model_chat_template(_model, IntPtr.Zero);

            if (templatePtr == IntPtr.Zero)
                return null;  // Model doesn't have a template

            return Marshal.PtrToStringUTF8(templatePtr);
        }

        /// <summary>
        /// Detect the chat format from the template
        /// </summary>
        public ChatFormat DetectChatFormat()
        {
            var template = GetChatTemplate();

            if (template == null)
                return ChatFormat.Unknown;

            // Check for known patterns
            if (template.Contains("<|im_start|>") && template.Contains("<|im_end|>"))
                return ChatFormat.ChatML;

            if (template.Contains("[INST]") && template.Contains("[/INST]"))
                return ChatFormat.Mistral;

            if (template.Contains("<<SYS>>") && template.Contains("<</SYS>>"))
                return ChatFormat.Llama2;

            if (template.Contains("<|user|>") && template.Contains("<|assistant|>"))
                return ChatFormat.Zephyr;

            if (template.Contains("<start_of_turn>"))
                return ChatFormat.Gemma;

            return ChatFormat.Unknown;
        }

        /// <summary>
        /// Evaluate a batch of tokens.
        /// </summary>
        private unsafe bool EvaluateBatch(int[] tokens)
        {
            fixed (int* tokensPtr = tokens)
            {
                var batch = Llama.llama_batch_get_one(tokensPtr, tokens.Length);
                int result = Llama.llama_decode(_context, batch);
                return result == 0;
            }
        }

        /// <summary>
        /// Get model information.
        /// </summary>
        public ModelInfo GetModelInfo()
        {
            int bufferSize = 1024;
            IntPtr buffer = Marshal.AllocHGlobal(bufferSize);

            try
            {
                Llama.llama_model_desc(_model, buffer, (UIntPtr)bufferSize);
                string description = Marshal.PtrToStringUTF8(buffer);

                return new ModelInfo
                {
                    Description = description,
                    ContextLength = Llama.llama_model_n_ctx_train(_model),
                    EmbeddingSize = Llama.llama_model_n_embd(_model),
                    LayerCount = Llama.llama_model_n_layer(_model),
                    ParameterCount = (long)Llama.llama_model_n_params(_model),
                    ModelSize = (long)Llama.llama_model_size(_model),
                    HasEncoder = Llama.llama_model_has_encoder(_model),
                    HasDecoder = Llama.llama_model_has_decoder(_model)
                };
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        // NEW: Helper methods for stop string detection
        private bool ContainsStopSequence(List<int> tokens, List<int[]> stopSequences)
        {
            foreach (var stopSeq in stopSequences)
            {
                if (tokens.Count < stopSeq.Length)
                    continue;

                // Check if the last N tokens match the stop sequence
                bool matches = true;
                for (int i = 0; i < stopSeq.Length; i++)
                {
                    if (tokens[tokens.Count - stopSeq.Length + i] != stopSeq[i])
                    {
                        matches = false;
                        break;
                    }
                }

                if (matches)
                    return true;
            }

            return false;
        }

        private void RemoveStopSequence(List<int> tokens, List<int[]> stopSequences)
        {
            foreach (var stopSeq in stopSequences)
            {
                if (tokens.Count < stopSeq.Length)
                    continue;

                // Check if tokens end with this stop sequence
                bool matches = true;
                for (int i = 0; i < stopSeq.Length; i++)
                {
                    if (tokens[tokens.Count - stopSeq.Length + i] != stopSeq[i])
                    {
                        matches = false;
                        break;
                    }
                }

                if (matches)
                {
                    // Remove the stop sequence
                    tokens.RemoveRange(tokens.Count - stopSeq.Length, stopSeq.Length);
                    return;
                }
            }
        }

        public void Dispose()
        {
            if (_disposed) return;

            if (_context != IntPtr.Zero)
            {
                Llama.llama_free(_context);
                _context = IntPtr.Zero;
            }

            if (_model != IntPtr.Zero)
            {
                Llama.llama_model_free(_model);
                _model = IntPtr.Zero;
            }

            Llama.llama_backend_free();
            _disposed = true;
        }
    }

    public class ChatMessage
    {
        public string Role { get; set; }     // "user", "assistant", "system"
        public string Content { get; set; }
    }


}