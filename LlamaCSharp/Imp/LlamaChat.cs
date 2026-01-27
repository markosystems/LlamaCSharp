using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace LlamaCSharp.Imp
{
    public class LlamaChat : IDisposable
    {
        public LlamaInference Model { get; private set; }
        public LlamaConfig Config { get; private set; }
        public GenerationConfig GenerationConfig { get; private set; }
        public ChatFormat Format { get; private set; }
        public bool ChatAlive { get; private set; }
        public int TotalTokensUsed { get; private set; }
        private int _maxHistoryTokens;

        public List<ChatMessage> ChatMessages { get; private set; }

       

        public LlamaChat(string modelPath,LlamaConfig llamaConfig = null, GenerationConfig generation = null,string systemPrompt = null)
        {
            Config = llamaConfig ?? new LlamaConfig();
            GenerationConfig = generation ?? new GenerationConfig()
            {
                MaxTokens = 150,
                RepeatPenaltyTokens = 64,   // Add defaults
                RepeatPenalty = 1.1f
            };

            // Setup silent mode BEFORE loading model
            if (Config.SilentMode)
            {
                Llama.llama_log_set((level, text, user_data) =>
                {
                    if (level <= -1)  // Only critical errors
                    {
                        Console.Error.WriteLine(Marshal.PtrToStringUTF8(text));
                    }
                }, IntPtr.Zero);
            }

            // Load model
            var modelParams = Llama.llama_model_default_params();
            modelParams.n_gpu_layers = Config.GpuLayers;
            modelParams.use_mmap = Config.UseMmap;
            modelParams.use_mlock = Config.UseMlock;

            var _model = Llama.llama_model_load_from_file(modelPath, modelParams);
            if (_model == IntPtr.Zero)
                throw new Exception($"Failed to load model from: {modelPath}");

            // Get model info
            var modelInfo = ModelInfo.GetModelInfo(_model);

            /// Use FULL context for KV cache
            Config.ContextSize = modelInfo.ContextLength;

            // Calculate history budget
            _maxHistoryTokens = Config.BatchSize - GenerationConfig.MaxTokens;

            // Initialize
            Model = new LlamaInference(_model, Config);
            ChatMessages = new List<ChatMessage>();
            Format = Model.DetectChatFormat();
            ChatAlive = true;
            TotalTokensUsed = 0;

            Console.WriteLine($"[Chat] Using {Config.ContextSize:N0} / {modelInfo.ContextLength:N0} tokens for history");
            Console.WriteLine($"[Chat] Format: {Format}");

            // Set system prompt if provided
            if (!string.IsNullOrEmpty(systemPrompt))
                SetSystemPrompt(systemPrompt);
        }
        public void AddStopStrings(string[] strings)
        {
            var ls = GenerationConfig.StopStrings.ToList();
            ls.AddRange(strings);
            GenerationConfig.StopStrings = ls.ToArray();
        }
        

        public void SetSystemPrompt(string prompt)
        {
            // Remove existing system message if any
            ChatMessages.RemoveAll(m => m.Role == "system");

            // Add new system message at the beginning
            ChatMessages.Insert(0, new ChatMessage { Role = "system", Content = prompt });
        }

        public string SendChat(string prompt, GenerationConfig generation = null, bool temp = false)
        {
            if (!ChatAlive)
                throw new InvalidOperationException("Chat is not alive. Create a new instance.");

            // Add user message
            ChatMessages.Add(new ChatMessage { Role = "user", Content = prompt });

            // Auto-trim to fit context
            TrimHistoryByTokens();

            // Format prompt
            var formattedPrompt = ChatFormation.FormatChat(ChatMessages, Format);

            var promptTokens = Model.Tokenize(formattedPrompt, addBos: true).Length;
            if(!Config.SilentMode)
            Console.WriteLine($"[Debug] Prompt tokens: {promptTokens}, Batch size: {Config.BatchSize}");

            if (promptTokens > Config.BatchSize)
            {
                Console.WriteLine($"[Warning] Prompt ({promptTokens}) exceeds batch size ({Config.BatchSize})!");
                // Auto-adjust or throw error
                
            }

            // Generate response
            var config = generation ?? GenerationConfig;
            var response = Model.Generate(formattedPrompt, config);
            response = CheckForSentenceEnd(response);
            

            // Update config if not temp
            if (generation != null && !temp)
                GenerationConfig = generation;

            // Track tokens
            TotalTokensUsed += EstimateTokens(prompt) + EstimateTokens(response);

            // Add assistant response
            ChatMessages.Add(new ChatMessage { Role = "assistant", Content = response });

            return response;
        }

        public async Task<string> SendChatAsync(
    string prompt,
    GenerationConfig generation = null,
    bool temp = false,
    CancellationToken cancellationToken = default)
        {
            if (!ChatAlive)
                throw new InvalidOperationException("Chat is not alive");

            // Add user message
            ChatMessages.Add(new ChatMessage { Role = "user", Content = prompt });

            // Auto-trim
            TrimHistoryByTokens();

            // Format prompt
            var formattedPrompt = ChatFormation.FormatChat(ChatMessages, Format);

            // Generate async
            var config = generation ?? GenerationConfig;
            var response = await Model.GenerateAsync(formattedPrompt, config, cancellationToken: cancellationToken);

            // Update config if not temp
            if (generation != null && !temp)
                GenerationConfig = generation;

            // Track tokens
            TotalTokensUsed += EstimateTokens(prompt) + EstimateTokens(response);

            // Add assistant response
            ChatMessages.Add(new ChatMessage { Role = "assistant", Content = response });

            return response;
        }

        string CheckForSentenceEnd(string f)
        {
            while (f[^1] != '.'&& f[^1] != '!' && f[^1] != '?')
            {
                f=f.Remove(f.Length-1, 1);
            }
            return f;
        }

        public void TrimHistoryByTokens()
        {
            if (ChatMessages.Count <= 1) return;

            var systemMessage = ChatMessages.FirstOrDefault(m => m.Role == "system");
            int totalTokens = systemMessage != null ? EstimateTokens(systemMessage.Content) : 0;

            var keepMessages = new List<ChatMessage>();
            if (systemMessage != null)
                keepMessages.Add(systemMessage);

            // Trim based on BATCH SIZE, not context size
            for (int i = ChatMessages.Count - 1; i >= 0; i--)
            {
                var msg = ChatMessages[i];
                if (msg.Role == "system") continue;

                var msgTokens = EstimateTokens(msg.Content);

                // Check against BATCH limit
                if (totalTokens + msgTokens > _maxHistoryTokens)
                    break;

                totalTokens += msgTokens;
                keepMessages.Insert(systemMessage != null ? 1 : 0, msg);
            }

            if (keepMessages.Count < ChatMessages.Count)
            {
                Console.WriteLine($"[Trim] Keeping {keepMessages.Count} messages (~{totalTokens} tokens, batch limit: {_maxHistoryTokens})");
            }

            ChatMessages.Clear();
            ChatMessages.AddRange(keepMessages);
        }

        private int EstimateTokens(string text)
        {
            return Model.Tokenize(text, addBos: false).Length;
        }

        public void KillChat()
        {
            ChatAlive = false;
            ChatMessages.Clear();
            Dispose();
        }

        public void Dispose()
        {
            Model?.Dispose();
        }
    }


}
