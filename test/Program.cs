using LlamaCSharp.Imp;
namespace test
{
    internal class Program
    {
        static void TrimHistory(List<ChatMessage> hist, int keepLast = 10)
        {
            if (hist.Count > keepLast + 1)  // +1 for system message
            {
                var systemMsg = hist[0];
                var recent = hist.Skip(hist.Count - keepLast).ToList();
                hist.Clear();
                hist.Add(systemMsg);
                hist.AddRange(recent);
            }
        }
        static void Main(string[] args)
        {
            LlamaConfig llamaConfig = new LlamaConfig();
            llamaConfig.SilentMode = true;
            var LLM = new LlamaInference(@"llamapp\mistral-7b-instruct-v0.2.Q4_K_M.gguf", llamaConfig, 0);
            string SystemPrompt = "You are a gym bro who only gives bad fitness advice";

            var history = new List<ChatMessage>();
            GenerationConfig config = new GenerationConfig
            {
                StopStrings = new[]
    {
        "</s>",          // Mistral's EOS token
        "[INST]",        // Stop if it tries to continue as user
        "\n\nUser:",     // Backup if format detection fails
    },
                MaxTokens = 300,     // Increased for full responses
                Temperature = 0.8f
            };

            string systemPrompt = "You are a gym bro influencer.\nYou love giving very bad fitness advice.";
            history.Add(new ChatMessage { Role = "system", Content = systemPrompt });

            Console.WriteLine("Chat with Gym Bro (type 'exit' to quit)\n");

            while (true)
            {
                Console.Write("You: ");
                string msg = Console.ReadLine();

                if (msg?.ToLower() == "exit") break;

                history.Add(new ChatMessage { Role = "user", Content = msg });

                var prompt = LLM.BuildMistralPrompt(history);
                // ↑ Don't print this - it's just formatting for the model

                var response = LLM.Generate(prompt, config);
                // ↑ This is what the AI actually said

                history.Add(new ChatMessage { Role = "assistant", Content = response });

                Console.WriteLine($"Gym Bro: {response}\n");
                // ↑ Only print the response

                TrimHistory(history);
            }
            LLM.Dispose();
        }
    }
}
