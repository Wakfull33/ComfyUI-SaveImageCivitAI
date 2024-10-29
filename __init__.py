from .save_civitai_node import SaveCivitai

NODE_CLASS_MAPPINGS = {
    "SaveCivitai" : SaveCivitai
}



NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveCivitai": "Save CivitAI",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']