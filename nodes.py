import torch
import PIL.Image as Image
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

    def aspect_ratio_to_size(self, aspect_ratio, resolution) -> tuple:
        resolution = ((int(resolution) + 63) // 64) * 64
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
        aspect_ratio_split = aspect_ratio.split(":")
        width_ratio = max(0, float(aspect_ratio_split[0]))
        height_ratio = max(0, float(aspect_ratio_split[1]))

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



class AspectRatio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ratio": (["16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "12:5", "1:1"], {"default": "16:9"}),
                "longer_side": ("INT", {"default": 1920, "min": 128, "max": 1024 * 16, "step": 64}),
            }
        }
    
    RETURN_NAMES = ("ratio", "ratio_w", "ratio_h", "width", "height", "longer_side", "shorter_side")
    RETURN_TYPES = ("STRING", "INT", "INT", "INT", "INT", "INT", "INT")
    
    def aspect_ratio_to_size(self, ratio, longer_side) -> tuple:
        longer_side = ((int(longer_side) + 63) // 64) * 64
        ratio_split = ratio.split(":")
        width_ratio = max(0, float(ratio_split[0]))
        height_ratio = max(0, float(ratio_split[1]))
        
        if width_ratio == 0 or height_ratio == 0:
            return (ratio, 0, 0, 0, 0, longer_side, 0)
        
        width = 0
        height = 0
        if width_ratio > height_ratio:
            width = longer_side
            height = int((longer_side / width_ratio) * height_ratio)
        else:
            height = longer_side
            width = int((longer_side / height_ratio) * width_ratio)
        
        if width < 0:
            width = 0
        if height < 0:
            height = 0
        
        return (ratio, width_ratio, height_ratio, width, height, longer_side, min(width, height))


class MatchImageToAspectRatio:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", ),
            },
            "optional": {
                "ratio_16_9": ("BOOLEAN", {"default": True}),
                "ratio_9_16": ("BOOLEAN", {"default": True}),
                "ratio_4_3": ("BOOLEAN", {"default": True}),
                "ratio_3_4": ("BOOLEAN", {"default": True}),
                "ratio_3_2": ("BOOLEAN", {"default": True}),
                "ratio_2_3": ("BOOLEAN", {"default": True}),
                "ratio_1_1": ("BOOLEAN", {"default": True}),
                "aspect_ratio": ("STRING", {"default": "16:9, 9:16"}),
            }
        }
    
    RETURN_NAMES = ("ratio", "ratio_w", "ratio_h")
    RETURN_TYPES = ("STRING", "INT", "INT")
    
    def find_ratio(self, image: torch.Tensor, ratio_list:list) -> tuple:
        
        # 画像の寸法を取得
        if isinstance(image, Image.Image):
            width, height = image.size()
        elif isinstance(image, torch.Tensor):
            width, height = image.shape[2], image.shape[1]
        
        # 実際の比率を計算
        actual_ratio = width / height
        
        # 最も近いアスペクト比を見つける
        min_diff = float('inf')
        closest_ratio = None, None
        
        for w, h in ratio_list:
            standard_ratio = w / h
            diff = abs(actual_ratio - standard_ratio)
            
            if diff < min_diff:
                min_diff = diff
                closest_ratio = (w, h)
        
        return closest_ratio

    def ratio_str_to_tuple(self, ratio_str:str) -> list:
        if not isinstance(ratio_str, str):
            return []
        if not ratio_str:
            return []
        ratio_str = ratio_str.strip()
        if not ratio_str:
            return []
        
        result = []

        if ";" in ratio_str:
            ratio_str = ratio_str.replace(";", ",")
        if "\n" in ratio_str:
            ratio_str = ratio_str.replace("\n", ",")
        if "." in ratio_str:
            ratio_str = ratio_str.replace(".", ",")
        if "/" in ratio_str:
            ratio_str = ratio_str.replace("/", ",")

        for ratio in ratio_str.split(","):
            if not ratio:
                continue
            ratio = ratio.strip()
            if not ratio:
                continue

            ratio_wh = ratio.split(":")
            if len(ratio_wh) != 2:
                continue
            
            try:
                w = int(ratio_wh[0].strip())
                h = int(ratio_wh[1].strip())
                if w <= 0 or h <= 0:
                    continue
                result.append((w, h))
            except:
                continue

        return result

    FUNCTION = "match_image_to_aspect_ratio"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def match_image_to_aspect_ratio(self, image: torch.Tensor, ratio_16_9:bool, ratio_9_16:bool, ratio_4_3:bool, ratio_3_4:bool, ratio_3_2:bool, ratio_2_3:bool, ratio_1_1:bool, aspect_ratio:str) -> Tuple[str, int, int]:
        
        # アスペクト比のリストを作成
        ratio_list = []
        if ratio_16_9:
            ratio_list.append((16, 9))
        if ratio_9_16:
            ratio_list.append((9, 16))
        if ratio_4_3:
            ratio_list.append((4, 3))
        if ratio_3_4:
            ratio_list.append((3, 4))
        if ratio_3_2:
            ratio_list.append((3, 2))
        if ratio_2_3:
            ratio_list.append((2, 3))
        if ratio_1_1:
            ratio_list.append((1, 1))
        
        ratio_list.extend(self.ratio_str_to_tuple(aspect_ratio))

        ratio_list = list(set(ratio_list))
        
        choise_w, choise_h = self.find_ratio(image, ratio_list)
        return (f"{choise_w}:{choise_h}", choise_w, choise_h)


class CalcFactorWidthHeight:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT",),
                "height": ("INT",),
                "factor": ("FLOAT",{"default": 1.5}),
                "divide": ("INT",{"default": 1, "step": 1, "min": 1}),
                "plus_divide": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_NAMES = ("width", "height", "large_side", "small_side", "width_float", "height_float", "large_side_float", "small_side_float")
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "FLOAT", "FLOAT")

    FUNCTION = "calc_width_height"
    OUTPUT_NODE = True

    CATEGORY = "image"

    def calc_width_height(
        self,
        width: int,
        height: int,
        factor: float,
        divide: int,
        plus_divide: bool
    ) -> Tuple[int, int, int, int, float, float, float, float]:
        """
        Scale (width, height) by *factor* and optionally snap the integer
        results to a multiple of *divide*.

        Parameters
        ----------
        width : int
            Original width in pixels.
        height : int
            Original height in pixels.
        factor : float
            Scaling factor (>0). A value of 0 is treated as invalid.
        divide : int
            Alignment unit. If >1, the integer results are rounded to the
            nearest multiple of this value. If <=1, no alignment is applied.
        plus_divide : bool
            Alignment direction when `divide` > 1  
            - False : round **down** (floor) to nearest multiple  
            - True  : round **up**  (ceil)  to nearest multiple

        Returns
        -------
        Tuple[int, int, int, int, float, float, float, float]
            (width_i, height_i, long_i, short_i,
             width_f, height_f, long_f, short_f)

            * `_f` … float results before alignment  
            * `_i` … int  results after alignment
        """
        # ── 0. 早期リターン ───────────────────────────
        if width == 0 or height == 0 or factor == 0 or divide == 0:
            return (0, 0, 0, 0, 0, 0, 0, 0)

        # 型を明確にそろえる
        width, height = int(width), int(height)
        factor, divide = float(factor), int(divide)

        # ── 1. スケーリング ───────────────────────────
        width_f, height_f = self._scale_dimensions(width, height, factor)
        long_f, short_f  = self._long_short(width_f, height_f)

        # ── 2. int 化 & 任意で倍数合わせ ──────────────
        width_i, height_i = int(width_f), int(height_f)

        if divide > 1:
            width_i  = self._align_to_divide(width_f, divide, plus_divide)
            height_i = self._align_to_divide(height_f, divide, plus_divide)

        _long_i, _short_i = self._long_short(width_i, height_i)
        long_i = int(_long_i)
        short_i = int(_short_i)

        return (
            width_i, height_i,
            long_i,  short_i,
            width_f, height_f,
            long_f,  short_f
        )

    # ────────────────── 内部ユーティリティ ──────────────────
    @staticmethod
    def _scale_dimensions(w: int, h: int, f: float) -> Tuple[float, float]:
        """Return (w * f, h * f) in float."""
        return w * f, h * f

    @staticmethod
    def _align_to_divide(value: float, div: int, ceil: bool) -> int:
        """
        Snap *value* to a multiple of *div*.
        Floor by default; if *ceil* is True, round up only when必要.
        """
        base = int(value / div) * div  # floor 相当
        if ceil and int(value) != base:
            base += div
        return base

    @staticmethod
    def _long_short(w: float, h: float) -> Tuple[float, float]:
        """Return (longer_side, shorter_side)."""
        return (max(w, h), min(w, h))


NODE_CLASS_MAPPINGS = {
    "AspectRatioToSize": AspectRatioToSize,
    "SizeToWidthHeight": SizeToWidthHeight,
    "CalculateImagePadding": CalculateImagePadding,
    "MatchImageToAspectRatio": MatchImageToAspectRatio,
    "CalcFactorWidthHeight": CalcFactorWidthHeight,
    "AspectRatio": AspectRatio,

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AspectRatioToSize": "AspectRatioToSize",
    "SizeToWidthHeight": "SizeToWidthHeight",
    "CalculateImagePadding": "CalculateImagePadding",
    "MatchImageToAspectRatio": "MatchImageToAspectRatio",
    "CalcFactorWidthHeight": "CalcFactorWidthHeight",
    "AspectRatio": "AspectRatio",
    
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

    import numpy as np

    def convert_to_tensor(image: Image.Image) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            return image
        if isinstance(image, np.ndarray):
            return torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        if isinstance(image, Image.Image):
            return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def gen_image(width, height):
        img = convert_to_tensor(Image.new("RGB", (width, height)))
        return img

    node = MatchImageToAspectRatio()
    rgb = gen_image(1920, 1080)
    print(node.match_image_to_aspect_ratio(rgb, True, True, True, True, True, True, True, "16:9, 9:16, 3:1"))
    print(node.match_image_to_aspect_ratio(rgb, False, True, True, True, True, True, True, "16:9, 9:16, 3:1"))
    print(node.match_image_to_aspect_ratio(rgb, False, True, True, True, True, True, True, "9:16, 3:1"))
    print(node.match_image_to_aspect_ratio(rgb, False, True, True, True, False, True, True, "9:16, 3:1"))
    print(node.match_image_to_aspect_ratio(rgb, False, True, False, True, False, True, True, "9:16, 3:1"))
    print(node.match_image_to_aspect_ratio(rgb, False, True, False, True, False, True, True, ""))

    print(node.match_image_to_aspect_ratio(gen_image(1000, 1000), True, True, True, True, True, True, True, "16:9, 9:16, 3:1"))
    print(node.match_image_to_aspect_ratio(gen_image(1000, 1001), True, True, True, True, True, True, True, "16:9, 9:16, 3:1"))
    print(node.match_image_to_aspect_ratio(gen_image(1000, 1010), True, True, True, True, True, True, True, "16:9, 9:16, 3:1"))
    print(node.match_image_to_aspect_ratio(gen_image(1000, 1100), True, True, True, True, True, True, True, "16:9, 9:16, 3:1"))
    print(node.match_image_to_aspect_ratio(gen_image(1000, 1500), True, True, True, True, True, True, True, "16:9, 9:16, 3:1"))

    node = CalcFactorWidthHeight()
    print(node.calc_width_height(1920, 1080, 1, 2, True))
    print(node.calc_width_height(1920, 1080, 1.5, 2, False))
    print(node.calc_width_height(1920, 1080, 2, 1, True))
    print(node.calc_width_height(1920, 1080, 3.141592653589793, 16, True))
    print(node.calc_width_height(1920, 1080, 3.141592653589793, 16, False))
    print(node.calc_width_height(1920, 1080, 3.141592653589793, 64, True))
    print(node.calc_width_height(1234, 4567, 3, 64, True))


#if __name__ == "__main__":
#    simple_test()
