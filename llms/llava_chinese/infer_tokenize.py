B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
simple_vision_conv_multimodal = {
    "system": "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
    "roles": {"human": "USER", "gpt": "ASSISTANT"},
}