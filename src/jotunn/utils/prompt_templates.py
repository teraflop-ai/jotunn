
def tagging_prompts(template, tags):
    
    if template == "intern":
        return f"""
        You must return ONLY the tags from the allowed tag list that truly apply to the image.

        Return the output ONLY in EXACT JSON format:

        {{
        "tags": ["tag1", "tag2"]
        }}

        Where each tag must be one of the allowed tags below and must NOT include anything else. There should only be one of each tag.

        Allowed tags: {tags}

        <image>
        """
    elif template == "qwen":
        return f"""
        <|im_start|>system
        You are a helpful assistant.<|im_end|>
        <|im_start|>user
        You must return ONLY the tags from the allowed tag list that truly apply to the image.

        Return the output ONLY in EXACT JSON format:

        {{
        "tags": ["tag1", "tag2"]
        }}

        Allowed tags: {tags}
        <|im_end|>
        <|im_start|>assistant
        <|vision_start|><|image_pad|><|vision_end|>
        """
    return ValueError()