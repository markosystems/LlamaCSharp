# LlamaCSharp
A really simple and lightweight implementation of Llama.cpp.

Before I started I didn't know about [LLama Sharp](https://github.com/SciSharp/LLamaSharp). Which is a more fleshed out implementation of LLama.cpp, and I recomend that you download that package for a more in-depth usage of models. 

## Design Philosophy

LlamaCSharp provides a simple foundation. Build what YOU need on top.

P/Invokes and Llama.cpp abstractions are in the base `LlamaCSharp` namespace.
My basic implementation is in the `LlamaCSharp.Imp` namespace.

### Implementation (namespace LlamaCSharp.Imp)
```csharp
var llm = new LlamaInference("model.gguf");
var output = llm.Generate("prompt");
```

### Your Extensions
Need chat? Build a ChatSession wrapper.  
Need streaming? Add a callback parameter.  
Need embeddings? Create an EmbeddingModel class.  
Need validation? Add your protocol layer.

The base stays simple. Your code adds complexity only where needed.
My implementation isn't gospel. feel free to improve upon it, or create your own.

### Why Not Include Everything?

Because then you'd be learning features you don't use.

LlamaCSharp is a **building block**, not a framework.
Like SDL. Like llama-cpp-python. 

Simple. Focused. Extensible.
