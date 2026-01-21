using System;
namespace LlamaCSharp.Imp
{
    
    public class GenerationConfig
    {
        public int MaxTokens { get; set; } = 512;
        public float Temperature { get; set; } = 0.7f;
        public int TopK { get; set; } = 40;
        public float TopP { get; set; } = 0.95f;
        public float MinP { get; set; } = 0.05f;
        public uint Seed { get; set; } = 0xFFFFFFFF; // Random seed
        public Action<int> OnTokenGenerated { get; set; }
        public string[] StopStrings { get; set; } = null;
    }
    
}