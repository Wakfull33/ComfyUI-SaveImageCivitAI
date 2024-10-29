import folder_paths

from PIL import Image
from PIL.PngImagePlugin import PngInfo

import numpy as np
import json
import os

from datetime import datetime

from .civitai_datas import GenerationData


class SaveCivitai:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""
        self.compress_level = 4

    methods = {"default": 4, "fastest": 0, "slowest": 6}

    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"images": ("IMAGE",),
                     "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                     "lora_stacker": ("STRING", {"default": "LoRA Stacker"}),
                     "efficient_loader": ("STRING", {"default": "Efficient Loader"}),
                     "sampler": ("STRING", {"default": "KSampler"}),
                     },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
                }

    RETURN_TYPES = ()

    FUNCTION = "save_images"

    OUTPUT_NODE = True

    CATEGORY = "MyTest"

    def process_metadata(self, lora_stacker, efficient_loader, sampler, datas):
        generated = GenerationData(datas, lora_stacker, efficient_loader, sampler)

        return generated.__str__()

    def save_images(self, images, filename_prefix="ComfyUI", lora_stacker="LoRA Stacker", efficient_loader="Efficient Loader", sampler="KSampler", prompt=None, extra_pnginfo=None):
        filename_prefix = datetime.now().strftime("%Y-%m-%d/") + filename_prefix + self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0])
        results = list()
        for (batch_number, image) in enumerate(images):
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            metadata = PngInfo()
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
                metadata.add_text("parameters", self.process_metadata(lora_stacker, efficient_loader, sampler, prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=self.compress_level)
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return {"ui": {"images": results}}


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "SaveCivitai": SaveCivitai
}
