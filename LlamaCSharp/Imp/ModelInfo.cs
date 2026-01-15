namespace LlamaCSharp.Imp
{
    public class ModelInfo
    {
        public string Description { get; set; }
        public int ContextLength { get; set; }
        public int EmbeddingSize { get; set; }
        public int LayerCount { get; set; }
        public long ParameterCount { get; set; }
        public long ModelSize { get; set; }
        public bool HasEncoder { get; set; }
        public bool HasDecoder { get; set; }

        public override string ToString()
        {
            return $"{Description}\n" +
                   $"Context: {ContextLength}, " +
                   $"Layers: {LayerCount}, " +
                   $"Params: {ParameterCount:N0}, " +
                   $"Size: {ModelSize / (1024 * 1024):N0} MB";
        }
    }
   
}