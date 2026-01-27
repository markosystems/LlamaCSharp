using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace LlamaCSharp.Imp
{
    public interface IInference
    {
        public LlamaInference inference { get; set; }
        public string Generate(string prompt, GenerationConfig config = null);
        public Task<string> GenerateAsync(string prompt, GenerationConfig config = null);

        public void Dispose();
    }
}
