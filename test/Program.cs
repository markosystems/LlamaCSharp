using LlamaCSharp.Imp;
namespace test
{
    internal class Program
    {
        static void Main(string[] args)
        {
            
            string SystemPrompt1 = "You're Batman, and you're talking to Superman. You want to confess that you have been sleeping with Lois Lane. Speak as Batman. Give graphic deatails.";
            string SystemPrompt2 = "You're Superman, and you're talking Batman. Speak as Superman. Lois is pregnant.";
            GenerationConfig BatConfig = new()
            {
                MaxTokens = 100,
                StopStrings = ["\n\n", "</s>"],
                Temperature = .99f
            };
            GenerationConfig SuperConfig = new()
            {
                MaxTokens = 100,
                StopStrings = ["\n\n", "</s>"],
                Temperature = .9f
            };
            var Batman = new LlamaChat(@"llamapp\mistral-7b-instruct-v0.2.Q4_K_M.gguf", generation: BatConfig, systemPrompt: SystemPrompt1);
            var Superman = new LlamaChat(@"llamapp\mistral-7b-instruct-v0.2.Q4_K_M.gguf", generation: SuperConfig, systemPrompt: SystemPrompt2);
            Console.WriteLine("Chat with Advice Bro (type 'exit' to quit)\n");
            string msg = "How are you Batman?";
            for (int i = 0; i < 3; i++)
            {
                var Batext = Batman.SendChat(msg);
                Console.WriteLine($"Batman: {Batext}\n");
                msg = Superman.SendChat(Batext);
                Console.WriteLine($"Superman: {msg}\n");
            }
            Batman.Dispose();
            Superman.Dispose();
        }
    }
}
