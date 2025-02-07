import torch
from typing import Tuple

class ResolutionSize:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __str__(self):
        return f"{self.width}x{self.height}"
    
    def __repr__(self):
        return f"{self.width}x{self.height}"
    
    def __eq__(self, other):
        return self.width == other.width and self.height == other.height


class SizeToWidthHeight:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "size": ("SIZE", ),
            }
        }
    
    RETURN_NAMES = ("Width", "Height", "LargeSide", "SmallSide")
    RETURN_TYPES = ("INT", "INT", "INT", "INT")

    FUNCTION = "size_to_width_height"
    OUTPUT_NODE = True

    def size_to_width_height(self, size) -> Tuple[int, int, int, int]:
        print(size, type(size))
        return (size.width, size.height, max(size.width, size.height), min(size.width, size.height), )


class AspectRatioToSize:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "aspect_ratio": ("STRING", {"default": "16:9"}),
                "resolution": ("INT", {"default": 1920, "min": 128, "max": 1024 * 8, "step": 64}),
            }
        }
    
    RETURN_NAMES = ("Size",)
    RETURN_TYPES = ("SIZE",)

    FUNCTION = "aspect_ratio_to_size"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def aspect_ratio_to_size(self, aspect_ratio, resolution) -> ResolutionSize:
        aspect_ratio = aspect_ratio.split(":")
        width_ratio = max(0, float(aspect_ratio[0]))
        height_ratio = max(0, float(aspect_ratio[1]))

        if width_ratio == 0 or height_ratio == 0:
            return (ResolutionSize(0, 0),)

        width = 0
        height = 0
        if width_ratio > height_ratio:
            width = resolution
            height = int(resolution / width_ratio * height_ratio)
        else:
            height = resolution
            width = int(resolution / height_ratio * width_ratio)
        
        if width < 0:
            width = 0
        if height < 0:
            height = 0

        resolution_size = ResolutionSize(width=width, height=height)

        return (resolution_size,)



class CalculateImagePadding:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
                "aspect_ratio": ("STRING", {"default": "16:9"}),
            }
        }
    
    RETURN_NAMES = ("left", "right", "top", "bottom")
    RETURN_TYPES = ("INT","INT","INT","INT")

    FUNCTION = "calculate_image_padding"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def calculate_image_padding(self, image: torch.Tensor, aspect_ratio:str) -> Tuple[int, int, int, int]:
        aspect_ratio = aspect_ratio.split(":")
        width_ratio = max(0, float(aspect_ratio[0]))
        height_ratio = max(0, float(aspect_ratio[1]))

        # Calculate the target aspect ratio
        target_ratio = width_ratio / height_ratio
        
        # Get image dimensions (assuming channel-first format: [C, H, W])
        height = image.shape[1]
        width = image.shape[2]

        # Calculate current aspect ratio
        current_ratio = width / height

        if current_ratio == target_ratio:
            # If aspect ratio matches, no padding is required
            return (0, 0, 0, 0)
        elif current_ratio > target_ratio:
            # Width is too large, add padding to the height (top and bottom)
            new_height = int(width / target_ratio)
            total_padding = new_height - height
            padding_top = total_padding // 2
            padding_bottom = total_padding - padding_top
            return (0, 0, padding_top, padding_bottom)
        else:
            # Height is too large, add padding to the width (left and right)
            new_width = int(height * target_ratio)
            total_padding = new_width - width
            padding_left = total_padding // 2
            padding_right = total_padding - padding_left
            return (padding_left, padding_right, 0, 0)


NODE_CLASS_MAPPINGS = {
    "AspectRatioToSize": AspectRatioToSize,
    "SizeToWidthHeight": SizeToWidthHeight,
    "CalculateImagePadding": CalculateImagePadding
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AspectRatioToSize": "AspectRatioToSize",
    "SizeToWidthHeight": "SizeToWidthHeight",
    "CalculateImagePadding": "CalculateImagePadding"
}


def simple_test():
    node = AspectRatioToSize()

    ratio = "16:9"
    result = node.aspect_ratio_to_size(ratio, 1920)
    print(ratio, result)

    ratio = "9:16"
    result = node.aspect_ratio_to_size(ratio, 1920)
    print(ratio, result)

    ratio = "4:3"
    result = node.aspect_ratio_to_size(ratio, 640)
    print(ratio, result)

    ratio = "3:4"
    result = node.aspect_ratio_to_size(ratio, 640)
    print(ratio, result)

    ratio = "1:1"
    result = node.aspect_ratio_to_size(ratio, 640)
    print(ratio, result)

    ratio = "1:0.5625"
    result = node.aspect_ratio_to_size(ratio, 1920)
    print(ratio, result)

    node = SizeToWidthHeight()
    size = ResolutionSize(1920, 1080)
    result = node.size_to_width_height(size)
    print(size, result)


#if __name__ == "__main__":
#    simple_test()
