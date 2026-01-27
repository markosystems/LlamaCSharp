using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;

namespace LlamaCSharp.Imp
{
    public static class ChatFormation
    {
        public static string FormatChat(List<ChatMessage> messages, ChatFormat format)
        {
            return format switch
            {
                ChatFormat.Mistral => BuildMistralPrompt(messages),
                ChatFormat.Llama2 => BuildLlama2Prompt(messages),
                ChatFormat.ChatML => BuildChatMLPrompt(messages),
                ChatFormat.Zephyr => BuildZephyrPrompt(messages),
                ChatFormat.Gemma => BuildGemmaPrompt(messages),
                _ => BuildPlainPrompt(messages)  // Fallback
            };
        }

        static private string BuildMistralPrompt(List<ChatMessage> ChatMessages)
        {
            var sb = new StringBuilder();
            var system = ChatMessages.FirstOrDefault(m => m.Role == "system");

            if (system != null)
            {
                sb.Append($"<s>[INST] {system.Content}\n\n");
            }
            else
            {
                sb.Append("<s>[INST] ");
            }

            for (int i = 1; i < ChatMessages.Count; i++)
            {
                var msg = ChatMessages[i];

                if (msg.Role == "user")
                {
                    if (i > 1) sb.Append("[INST] ");
                    sb.Append(msg.Content);
                    sb.Append(" [/INST]");
                }
                else if (msg.Role == "assistant")
                {
                    sb.Append($" {msg.Content}</s>");
                }
            }

            return sb.ToString();
        }

        static private string BuildChatMLPrompt(List<ChatMessage> messages)
        {
            var sb = new StringBuilder();

            foreach (var msg in messages)
            {
                sb.AppendLine($"<|im_start|>{msg.Role}");
                sb.AppendLine(msg.Content);
                sb.AppendLine("<|im_end|>");
            }

            sb.Append("<|im_start|>assistant\n");
            return sb.ToString();
        }
        static private string BuildGemmaPrompt(List<ChatMessage> messages)
        {
            var sb = new StringBuilder();

            foreach (var msg in messages)
            {
                sb.AppendLine($"<start_of_turn>{msg.Role}");
                sb.AppendLine(msg.Content);
                sb.AppendLine("<end_of_turn>");
            }

            sb.Append("<start_of_turn>assistant\n");
            return sb.ToString();
        }
        static private string BuildZephyrPrompt(List<ChatMessage> messages)
        {
            var sb = new StringBuilder();

            foreach (var msg in messages)
            {
                sb.AppendLine($"<|{msg.Role}|>");
                sb.AppendLine(msg.Content);
            }

            sb.Append("<|im_start|>assistant\n");
            return sb.ToString();
        }

        static private string BuildLlama2Prompt(List<ChatMessage> messages)
        {
            var sb = new StringBuilder();
            var system = messages.FirstOrDefault(m => m.Role == "system");

            sb.Append("<s>[INST] ");

            if (system != null)
            {
                sb.Append($"<<SYS>>\n{system.Content}\n<</SYS>>\n\n");
            }

            for (int i = 1; i < messages.Count; i++)
            {
                var msg = messages[i];

                if (msg.Role == "user")
                {
                    if (i > 1) sb.Append("<s>[INST] ");
                    sb.Append(msg.Content);
                    sb.Append(" [/INST]");
                }
                else if (msg.Role == "assistant")
                {
                    sb.Append($" {msg.Content} </s>");
                }
            }

            return sb.ToString();
        }

        static private string BuildPlainPrompt(List<ChatMessage> messages)
        {
            // Fallback for models without special formatting
            var sb = new StringBuilder();

            foreach (var msg in messages)
            {
                if (msg.Role == "system")
                    sb.AppendLine($"System: {msg.Content}\n");
                else if (msg.Role == "user")
                    sb.AppendLine($"User: {msg.Content}");
                else if (msg.Role == "assistant")
                    sb.AppendLine($"Assistant: {msg.Content}");
            }

            sb.Append("Assistant:");
            return sb.ToString();
        }

        public static string GetChatTemplate(IntPtr _model)
        {
            var templatePtr = Llama.llama_model_chat_template(_model, IntPtr.Zero);

            if (templatePtr == IntPtr.Zero)
                return null;  // Model doesn't have a template

            return Marshal.PtrToStringUTF8(templatePtr);
        }
    }

    public enum ChatFormat
    {
        Unknown,
        Mistral,    // <s>[INST] ... [/INST]
        Llama2,     // <<SYS>> ... <</SYS>>
        ChatML,     // <|im_start|> ... <|im_end|>
        Zephyr,     // <|user|> ... <|assistant|>
        Gemma,      // <start_of_turn>user ... <end_of_turn>
    }
}
