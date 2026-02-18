# python -m unittest test_nodes.py

import unittest
import os
import json
from nodes import AspectRatioToSize, SizeToWidthHeight, MatchImageToAspectRatio, CalcFactorWidthHeight, AspectRatio, CalculateImagePadding, ResolutionSize
from PIL import Image
import torch
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


class TestAspectRatioToSize(unittest.TestCase):
    def setUp(self):
        pass

    def test_tag_filter(self):
        node = AspectRatioToSize()

        ratio = "16:9"
        result = node.aspect_ratio_to_size(ratio, 1920)[0]
        self.assertEqual(result.width, 1920)
        self.assertEqual(result.height, 1080)

        ratio = "9:16"
        result = node.aspect_ratio_to_size(ratio, 1920)[0]
        self.assertEqual(result.width, 1080)
        self.assertEqual(result.height, 1920)

        ratio = "4:3"
        result = node.aspect_ratio_to_size(ratio, 640)[0]
        self.assertEqual(result.width, 640)
        self.assertEqual(result.height, 480)

        ratio = "3:4"
        result = node.aspect_ratio_to_size(ratio, 640)[0]
        self.assertEqual(result.width, 480)
        self.assertEqual(result.height, 640)

        ratio = "1:1"
        result = node.aspect_ratio_to_size(ratio, 640)[0]
        self.assertEqual(result.width, 640)
        self.assertEqual(result.height, 640)

        ratio = "1:0.5625"
        result = node.aspect_ratio_to_size(ratio, 1920)[0]
        self.assertEqual(result.width, 1920)
        self.assertEqual(result.height, 1080)

    def test_size_to_width_height(self):
        node = SizeToWidthHeight()
        size = ResolutionSize(1920, 1080)
        result = node.size_to_width_height(size)
        self.assertEqual(result, (1920, 1080, 1920, 1080))

    def test_match_image_to_aspect_ratio(self):
        node = MatchImageToAspectRatio()
        rgb = gen_image(1920, 1080)
        self.assertEqual(node.match_image_to_aspect_ratio(rgb, True, True, True, True, True, True, True, "16:9, 9:16, 3:1"), ("16:9", 16, 9))
        self.assertEqual(node.match_image_to_aspect_ratio(rgb, False, True, True, True, True, True, True, "16:9, 9:16, 3:1"), ("16:9", 16, 9))
        self.assertEqual(node.match_image_to_aspect_ratio(rgb, False, True, True, True, True, True, True, "9:16, 3:1"), ("3:2", 3, 2))
        self.assertEqual(node.match_image_to_aspect_ratio(rgb, False, True, True, True, False, True, True, "9:16, 3:1"), ("4:3", 4, 3))
        self.assertEqual(node.match_image_to_aspect_ratio(rgb, False, True, False, True, False, True, True, "9:16, 3:1"), ("1:1", 1, 1))
        self.assertEqual(node.match_image_to_aspect_ratio(rgb, False, True, False, True, False, True, True, ""), ("1:1", 1, 1))

        self.assertEqual(node.match_image_to_aspect_ratio(gen_image(1000, 1000), True, True, True, True, True, True, True, "16:9, 9:16, 3:1"), ("1:1", 1, 1))
        self.assertEqual(node.match_image_to_aspect_ratio(gen_image(1000, 1001), True, True, True, True, True, True, True, "16:9, 9:16, 3:1"), ("1:1", 1, 1))
        self.assertEqual(node.match_image_to_aspect_ratio(gen_image(1000, 1010), True, True, True, True, True, True, True, "16:9, 9:16, 3:1"), ("1:1", 1, 1))
        self.assertEqual(node.match_image_to_aspect_ratio(gen_image(1000, 1100), True, True, True, True, True, True, True, "16:9, 9:16, 3:1"), ("1:1", 1, 1))
        self.assertEqual(node.match_image_to_aspect_ratio(gen_image(1000, 1500), True, True, True, True, True, True, True, "16:9, 9:16, 3:1"), ("2:3", 2, 3))

    def test_calc_factor_width_height(self):
        node = CalcFactorWidthHeight()
        # return ("width", "height", "large_side", "small_side", "width_float", "height_float", "large_side_float", "small_side_float")
        self.assertEqual(node.calc_width_height(1920, 1080, 1, 2, True), (1920, 1080, 1920, 1080, 1920.0, 1080.0, 1920.0, 1080.0))
        self.assertEqual(node.calc_width_height(1920, 1080, 1.5, 2, False), (2880, 1620, 2880, 1620, 2880.0, 1620.0, 2880.0, 1620.0))
        self.assertEqual(node.calc_width_height(1920, 1080, 2, 1, True), (3840, 2160, 3840, 2160, 3840.0, 2160.0, 3840.0, 2160.0))
        self.assertEqual(node.calc_width_height(1920, 1080, 3, 16, True), (5760, 3248, 5760, 3248, 5760.0, 3240.0, 5760.0, 3240.0))
        self.assertEqual(node.calc_width_height(1920, 1080, 0.5, 16, False), (960, 528, 960, 528, 960.0, 540.0, 960.0, 540.0))
        self.assertEqual(node.calc_width_height(1234, 4567, 3, 64, True), (3712, 13760, 13760, 3712, 3702.0, 13701.0, 13701.0, 3702.0))


    def test_calculate_image_padding(self):
        node = AspectRatio()
        self.assertEqual(node.aspect_ratio_to_size("16:9", 1920), ('16:9', 16.0, 9.0, 1920, 1080, 1920, 1080))
        self.assertEqual(node.aspect_ratio_to_size("9:16", 1920), ('9:16', 9.0, 16.0, 1080, 1920, 1920, 1080))
        self.assertEqual(node.aspect_ratio_to_size("4:3", 640), ('4:3', 4.0, 3.0, 640, 480, 640, 480))
        self.assertEqual(node.aspect_ratio_to_size("3:4", 640), ('3:4', 3.0, 4.0, 480, 640, 640, 480))
        self.assertEqual(node.aspect_ratio_to_size("1:1", 640), ('1:1', 1.0, 1.0, 640, 640, 640, 640))
        self.assertEqual(node.aspect_ratio_to_size("1:0.5625", 1920), ('1:0.5625', 1.0, 0.5625, 1920, 1080, 1920, 1080))

    def test_calculate_image_padding(self):
        node = CalculateImagePadding()
        # return ("left", "right", "top", "bottom")
        rgb = gen_image(1600, 900)
        self.assertEqual(node.calculate_image_padding(rgb, "16:9"), (0, 0, 0, 0))
        self.assertEqual(node.calculate_image_padding(rgb, "9:16"), (0, 0, 972, 972))
        self.assertEqual(node.calculate_image_padding(rgb, "4:3"), (0, 0, 150, 150))
        self.assertEqual(node.calculate_image_padding(rgb, "3:4"), (0, 0, 616, 617))
        self.assertEqual(node.calculate_image_padding(rgb, "1:1"), (0, 0, 350, 350))
        self.assertEqual(node.calculate_image_padding(rgb, "1:0.5625"), (0, 0, 0, 0))


if __name__ == '__main__':
    unittest.main()
