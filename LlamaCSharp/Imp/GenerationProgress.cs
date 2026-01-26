using System;
using System.Collections.Generic;
using System.Text;

namespace LlamaCSharp.Imp
{
    public class GenerationProgress
    {
        public int CurrentChunk { get; set; }
        public int TotalChunks { get; set; }
        public int PercentComplete { get; set; }
        public string Status { get; set; }
        public string LatestChunk { get; set; }
        public bool IsComplete { get; set; }
    }
}
