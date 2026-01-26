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
        public uint Seed { get; set; } = 0xFFFFFFFF;
        public Action<int> OnTokenGenerated { get; set; }
        public string[] StopStrings { get; set; } = null;

        public int RepeatPenaltyTokens { get; set; } = 64;      // Last N tokens to penalize (0 = disabled)
        public float RepeatPenalty { get; set; } = 1.0f;        // 1.0 = no penalty, >1.0 = penalize repeats
        public float FrequencyPenalty { get; set; } = 0.0f;     // 0.0 = disabled
        public float PresencePenalty { get; set; } = 0.0f;
    }

}