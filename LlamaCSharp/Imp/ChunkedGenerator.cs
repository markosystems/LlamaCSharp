using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace LlamaCSharp.Imp
{
    public class ChunkedGenerator
    {
        private LlamaInference _llm;
        public string SystemPrompt { get; set; }

        public string[] Transitions = new[]
        {
            "I would also add,",
            "It is worth noting,",
            "One must also remember that",
            "It bears repeating that",
            "Perhaps unnecessarily,",
            "At the risk of tedium,",
            "Incidentally,",
            "More to the point,",
            "This is compounded by the fact that",
            "Which is to say,",
            "And this is before considering how"
        };

        public ChunkedGenerator(LlamaInference llm, string systemPrompt = "")
        {
            _llm = llm;
            SystemPrompt = systemPrompt;
        }

        public string Generate(string prompt, int chunks = 10, GenerationConfig config = null, int wordcontext = 20)
        {
            config ??= new GenerationConfig
            {
                MaxTokens = 200,
                Temperature = 0.7f,
                RepeatPenaltyTokens = 128,
                RepeatPenalty = 1.15f,
                FrequencyPenalty = 0.3f
            };

            var fullResponse = new StringBuilder();
            var context = $"{SystemPrompt}\n{prompt}";

            for (int i = 0; i < chunks; i++)
            {
                Console.WriteLine($"[Chunk {i + 1}/{chunks}]");

                // Generate chunk
                var chunk = _llm.Generate(context, config);

                // Clean up incomplete sentences
                chunk = RemoveIncompleteSentence(chunk);

                if (string.IsNullOrWhiteSpace(chunk))
                    break;  // Stop if we got nothing useful

                // Add to full response
                fullResponse.Append(chunk);
                fullResponse.Append(" ");

                // Build context for next chunk
                var lastWords = GetLastWords(chunk, wordCount: wordcontext);
                var transition = GetRandomTransition();

                context = $"{SystemPrompt}\n{transition}\n{lastWords}";
            }

            return fullResponse.ToString().Trim();
        }

        public event EventHandler<GenerationProgress> OnProgress;

        // Async version
        public async Task<string> GenerateAsync(
            string prompt,
            int chunks = 10,
            GenerationConfig config = null,
            IProgress<GenerationProgress> progress = null,
            CancellationToken cancellationToken = default,
            int wordcontext = 20)
        {
            config ??= new GenerationConfig { MaxTokens = 200, Temperature = 0.7f };

            var fullResponse = new StringBuilder();
            var context = $"{SystemPrompt}\n{prompt}";

            for (int i = 0; i < chunks; i++)
            {
                // Check for cancellation
                cancellationToken.ThrowIfCancellationRequested();

                // Report progress
                progress?.Report(new GenerationProgress
                {
                    CurrentChunk = i + 1,
                    TotalChunks = chunks,
                    PercentComplete = (int)((i / (float)chunks) * 100),
                    Status = $"Generating chunk {i + 1}/{chunks}..."
                });

                // Generate chunk (run on thread pool to not block UI)
                var chunk = await Task.Run(() =>
                    _llm.Generate(context, config),
                    cancellationToken
                );

                chunk = RemoveIncompleteSentence(chunk);

                if (string.IsNullOrWhiteSpace(chunk))
                    break;

                fullResponse.Append(chunk);
                fullResponse.Append(" ");

                // Report chunk complete with text
                progress?.Report(new GenerationProgress
                {
                    CurrentChunk = i + 1,
                    TotalChunks = chunks,
                    PercentComplete = (int)(((i + 1) / (float)chunks) * 100),
                    Status = $"Chunk {i + 1} complete",
                    LatestChunk = chunk
                });

                var lastWords = GetLastWords(chunk, wordcontext);
                var transition = GetRandomTransition();
                context = $"{SystemPrompt}\n{transition}\n{lastWords}";
            }

            // Final progress
            progress?.Report(new GenerationProgress
            {
                CurrentChunk = chunks,
                TotalChunks = chunks,
                PercentComplete = 100,
                Status = "Complete!",
                IsComplete = true
            });

            return fullResponse.ToString().Trim();
        }

        public string RemoveIncompleteSentence(string text)
        {
            text = text.TrimEnd();

            if (string.IsNullOrEmpty(text))
                return text;

            // Already ends with proper punctuation
            if (text.EndsWith(".") || text.EndsWith("!") || text.EndsWith("?"))
                return text;

            // Find last sentence ender
            int lastPeriod = text.LastIndexOf('.');
            int lastExclaim = text.LastIndexOf('!');
            int lastQuestion = text.LastIndexOf('?');

            int lastEnder = Math.Max(lastPeriod, Math.Max(lastExclaim, lastQuestion));

            // If found, truncate to last complete sentence
            if (lastEnder > 0)
                return text.Substring(0, lastEnder + 1);

            // No sentence enders found, return as-is (probably first chunk)
            return text;
        }

        public string GetLastWords(string text, int wordCount = 20)
        {
            var words = text.Split(new[] { ' ', '\n', '\r' },
                StringSplitOptions.RemoveEmptyEntries);

            if (words.Length <= wordCount)
                return text;

            var lastWords = words.Skip(words.Length - wordCount).ToArray();
            return string.Join(" ", lastWords);
        }

        public string GetRandomTransition()
        {
            var rng = new Random();
            return Transitions[rng.Next(Transitions.Length)];
        }

        public void SetTransitions(string[] transitions)=>Transitions = transitions;
    }

    
}
