using System;
using System.Collections.Generic;
using System.Text;

namespace LlamaCSharp
{
   

    public enum llama_vocab_type
    {
        LLAMA_VOCAB_TYPE_NONE = 0,
        LLAMA_VOCAB_TYPE_SPM = 1,
        LLAMA_VOCAB_TYPE_BPE = 2,
        LLAMA_VOCAB_TYPE_WPM = 3,
        LLAMA_VOCAB_TYPE_UGM = 4,
        LLAMA_VOCAB_TYPE_RWKV = 5,
        LLAMA_VOCAB_TYPE_PLAMO2 = 6,
    }

    public enum llama_ftype
    {
        LLAMA_FTYPE_ALL_F32 = 0,
        LLAMA_FTYPE_MOSTLY_F16 = 1,
        LLAMA_FTYPE_MOSTLY_Q4_0 = 2,
        LLAMA_FTYPE_MOSTLY_Q4_1 = 3,
        LLAMA_FTYPE_MOSTLY_Q8_0 = 7,
        LLAMA_FTYPE_MOSTLY_Q5_0 = 8,
        LLAMA_FTYPE_MOSTLY_Q5_1 = 9,
    }

    public enum llama_pooling_type
    {
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
        LLAMA_POOLING_TYPE_NONE = 0,
        LLAMA_POOLING_TYPE_MEAN = 1,
        LLAMA_POOLING_TYPE_CLS = 2,
        LLAMA_POOLING_TYPE_LAST = 3,
        LLAMA_POOLING_TYPE_RANK = 4,
    }

    public enum llama_split_mode
    {
        LLAMA_SPLIT_MODE_NONE = 0,
        LLAMA_SPLIT_MODE_LAYER = 1,
        LLAMA_SPLIT_MODE_ROW = 2,
    }

}
