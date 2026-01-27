using System;
using System.Runtime.InteropServices;

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

        public static ModelInfo GetModelInfo(IntPtr model)
        {
            int bufferSize = 1024;
            IntPtr buffer = Marshal.AllocHGlobal(bufferSize);

            try
            {
                Llama.llama_model_desc(model, buffer, (UIntPtr)bufferSize);
                string description = Marshal.PtrToStringUTF8(buffer);

                return new ModelInfo
                {
                    Description = description,
                    ContextLength = Llama.llama_model_n_ctx_train(model),
                    EmbeddingSize = Llama.llama_model_n_embd(model),
                    LayerCount = Llama.llama_model_n_layer(model),
                    ParameterCount = (long)Llama.llama_model_n_params(model),
                    ModelSize = (long)Llama.llama_model_size(model),
                    HasEncoder = Llama.llama_model_has_encoder(model),
                    HasDecoder = Llama.llama_model_has_decoder(model)
                };
            }
            finally
            {
                Marshal.FreeHGlobal(buffer);
            }
        }

        public override string ToString()
        {
            return $"{Description}\n" +
                   $"Context: {ContextLength:N0}, \n" +
                   $"Layers: {LayerCount}, \n" +
                   $"Params: {ParameterCount:N0}, \n" +
                   $"Size: {ModelSize / (1024 * 1024):N0} MB";
        }
    }
   
}