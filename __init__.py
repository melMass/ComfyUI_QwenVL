from .nodes import *

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Qwen2.5": Qwen2VL,
    "LoadQwenLModel": QwenLoadLModel,
    "QwenOmni": QwenOmni,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "Qwen2.5": "Qwen2.5",
    "Load Qwen Model": "LoadQwenLModel",
    "Qwen Omni": "QwenOmni",
}
