using LlamaCSharp.Imp;
using LlamaCSharp.Inference;
namespace test
{
    internal class Program
    {
        static void Main(string[] args)
        {
            LlamaConfig llamaConfig = new LlamaConfig();
            llamaConfig.SilentMode = true;
            var LLM = new LlamaInference(@"llamapp\mistral-7b-instruct-v0.2.Q4_K_M.gguf");
            string SystemPrompt = "You are my bestest homie in the world.";

            GenerationConfig config = new GenerationConfig();
            config.MaxTokens = 150;
            Console.WriteLine("Write something?");
            string Msg = Console.ReadLine();
            var responsre = LLM.Generate(Msg, config);
            Console.WriteLine(responsre);
            LLM.Dispose();
        }
    }
}
