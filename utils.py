import torch
import gc
import json
import re

def clear_cuda_memory(model_name):
    """Clears a specific model from cache and frees GPU memory."""
    global _kani_cache, _hf_engine_cache

    if model_name in _kani_cache:
        del _kani_cache[model_name]

    if model_name in _hf_engine_cache:
        del _hf_engine_cache[model_name]
    gc.collect()

    # Tell PyTorch to release unused cached memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def repair_json(content):
    
    # Clean up the content
    content = _remove_think_tags(content)
    content = _extract_json_from_codeblock(content)
    content = _fix_triple_quoted_strings(content)
    content = _extract_json_from_plain_text(content)
    content = _fix_unescaped_characters(content)
    content = _strip_formatting(content)

    return content

def _remove_think_tags(text):
    think_end = text.find('</think>')
    if think_end != -1:
        return text[think_end + 8:].strip()  # len('</think>') = 8
    return text

def _extract_json_from_codeblock(text):
    # Try to match ```json ... ``` first
    patterns = [
        r"(?s)```json\s*(.*?)\s*```",  # JSON code block
        r"(?s)```\s*(.*?)\s*```"        # Generic code block
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    
    return text

def _extract_json_from_plain_text(text: str) -> str:
    last_brace_pos = text.rfind('{')
    if last_brace_pos != -1:
        return text[last_brace_pos:]

    return text

def _fix_triple_quoted_strings(text):
    pattern = r'"(\w+)":\s*"""(.*?)"""'
    
    def escape_for_json(match):
        key = match.group(1)
        content = match.group(2)
        
        # Escape special characters for JSON
        replacements = [
            ('\\', '\\\\'),  # Backslashes first
            ('"', '\\"'),    # Quotes
            ('\n', '\\n'),   # Newlines
            ('\r', '\\r'),   # Carriage returns
            ('\t', '\\t'),   # Tabs
        ]
        
        for old, new in replacements:
            content = content.replace(old, new)
        
        return f'"{key}": "{content}"'
    
    return re.sub(pattern, escape_for_json, text, flags=re.DOTALL)

def _fix_unescaped_characters(text: str) -> str:
    # Set a limit to prevent infinite loops on unfixable errors

    text = text.replace('\\\n', '\\n') # llama

    max_attempts = 200
    
    for i in range(max_attempts):
        try:
            # Try to parse the text. If it works, we're done.
            json.loads(text)
            return text
        except json.JSONDecodeError as e:
            # On error, the parser tells us exactly where the problem is.
            # e.pos is the character index of the error.
            
            # Case 1: An unescaped newline or other control character in a string
            if "Invalid control character" in e.msg:
                # Replace the problematic character with its escaped version
                char_to_escape = text[e.pos]
                if char_to_escape == '\n':
                    escaped_char = '\\n'
                elif char_to_escape == '\r':
                    escaped_char = '\\r'
                elif char_to_escape == '\t':
                    escaped_char = '\\t'
                else:
                    # If it's some other control character, just remove it
                    escaped_char = ''
                
                text = text[:e.pos] + escaped_char + text[e.pos + 1:]

            # Case 2: An unescaped double quote in a string
            elif "Unterminated string" in e.msg:
                # This often means a '"' is inside a string without being escaped.
                # We'll try to find the quote just before the error and escape it.
                text = text[:e.pos - 1] + '\\' + text[e.pos - 1:]

            # If we can't fix it, break the loop and return the broken text
            else:
                break
    return text

def _strip_formatting(text):
    # Remove leading/trailing markdown json markers
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]  # len("```json") = 7
    if text.endswith("```"):
        text = text[:-3]
    
    # Remove surrounding quotes if present
    text = text.strip()
    if text.startswith("'") and text.endswith("'") and len(text) > 1:
        text = text[1:-1]
    
    return text.strip()


