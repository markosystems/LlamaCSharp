using System;

namespace LlamaCSharp.Imp
{
    

    public class LlamaConfig
    {
        public int ContextSize { get; set; } = 4096;
        public int BatchSize { get; set; } = 512;
        public int Threads { get; set; } = 8;
        public int ThreadsBatch { get; set; } = 8;
        public int GpuLayers { get; set; } = 0;
        public bool UseMmap { get; set; } = true;
        public bool UseMlock { get; set; } = false;
        public bool SilentMode { get; set; } = true;
    }
    
}