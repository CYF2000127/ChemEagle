# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random
import math

import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F

import numpy as np


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def crop(image, target, region):
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # should we do something wrt the original size?
    # target["size"] = torch.tensor([h, w])

    fields = ["labels", "area"]

    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    # remove elements for which the boxes or masks that have zero area
    # if "boxes" in target or "masks" in target:
    #     # favor boxes selection when defining which elements to keep
    #     # this is compatible with previous implementation
    #     if "boxes" in target:
    #         cropped_boxes = target['boxes'].reshape(-1, 2, 2)
    #         keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
    #     else:
    #         keep = target['masks'].flatten(1).any(1)
    #
    #     for field in fields:
    #         target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    return flipped_image, target


def rotate90(image, target):
    rotated_image = image.rotate(90, expand=1)

    w, h = rotated_image.size

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [1, 2, 3, 0]] * torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor([0, h, 0, h])
        target["boxes"] = boxes

    return rotated_image, target


def resize(image, target, size, max_size=None):
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    return rescaled_image, target


def pad(image, target, padding):
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None
    target = target.copy()
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))
    return padded_image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: dict):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomReactionCrop(object):
    def __init__(self):
        pass

    def __call__(self, img, target):
        w, h = img.size
        boxes = target["boxes"]
        x_avail = [1] * w
        y_avail = [1] * h
        for reaction in target['reactions']:
            ids = reaction['reactants'] + reaction['conditions'] + reaction['products']
            rboxes = boxes[ids].round().int()
            rmin, _ = rboxes.min(dim=0)
            rmax, _ = rboxes.max(dim=0)
            x1, x2 = (rmin[0].item(), rmax[2].item())
            for i in range(x1, x2):
                x_avail[i] = 0
            y1, y2 = (rmin[1].item(), rmax[3].item())
            for i in range(y1, y2):
                y_avail[i] = 0

        def sample_from_avail(w):
            spans = []
            left, right = 0, 0
            while right < len(w):
                while right < len(w) and w[left] == w[right]:
                    right += 1
                if w[left] == 1:
                    spans.append((left, right))
                left, right = right + 1, right + 1
            if w[0] == 0:
                spans = [(0, 0)] + spans
            if w[-1] == 0:
                spans = spans + [(len(w), len(w))]
            if len(spans) < 2:
                w1 = random.randint(0, len(w))
                w2 = random.randint(0, len(w))
            else:
                spans = random.sample(spans, 2)
                w1 = random.randint(*spans[0])
                w2 = random.randint(*spans[1])
            return min(w1, w2), max(w1, w2)

        x1, x2 = sample_from_avail(x_avail)
        y1, y2 = sample_from_avail(y_avail)
        region = (y1, x1, y2-y1, x2-x1)
        if x2-x1 < 30 or y2-y1 < 30:
            # Cropped region too small
            return img, target
        else:
            return crop(img, target, region)


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomRotate(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return rotate90(img, target)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target=None):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class Resize(object):
    def __init__(self, size):
        assert isinstance(size, (list, tuple))
        self.size = size

    def __call__(self, img, target=None):
        return resize(img, target, self.size)


class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std, debug=False):
        self.mean = mean
        self.std = std
        self.debug = debug

    def __call__(self, image, target=None):
        if not self.debug:
            image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, None
        target = target.copy()
        h, w = image.shape[-2:]
        if "boxes" in target:
            boxes = target["boxes"]
            boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
            target["boxes"] = boxes.clamp(min=0, max=1)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string


class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
    """

    def __init__(self, output_size=1333, aug_scale_min=0.3, aug_scale_max=2.0):
        self.desired_size = output_size
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.random = (aug_scale_min != 1) or (aug_scale_max != 1)

    def rescale_target(self, scaled_size, image_size, target):
        # compute rescaled targets
        image_scale = scaled_size / image_size
        ratio_height, ratio_width = image_scale

        target = target.copy()

        if "boxes" in target:
            boxes = target["boxes"]
            scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
            target["boxes"] = scaled_boxes

        if "area" in target:
            area = target["area"]
            scaled_area = area * (ratio_width * ratio_height)
            target["area"] = scaled_area

        return target

    def crop_target(self, region, target):
        i, j, h, w = region
        fields = ["labels", "area"]

        target = target.copy()

        if "boxes" in target:
            boxes = target["boxes"]
            max_size = torch.as_tensor([w, h], dtype=torch.float32)
            cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
            cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
            cropped_boxes = cropped_boxes.clamp(min=0)
            area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
            target["boxes"] = cropped_boxes.reshape(-1, 4)
            target["area"] = area
            fields.append("boxes")

        # Do not remove the boxes with zero area. Tokenizer does it instead.
        # if "boxes" in target:
        #     # favor boxes selection when defining which elements to keep
        #     # this is compatible with previous implementation
        #     cropped_boxes = target['boxes'].reshape(-1, 2, 2)
        #     keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        #     for field in fields:
        #         target[field] = target[field][keep]
        return target

    def pad_target(self, padding, target):
        # padding: left, top, right, bottom
        target = target.copy()
        if "boxes" in target:
            left, top, right, bottom = padding
            target["boxes"][:, 0::2] += left
            target["boxes"][:, 1::2] += top
        return target

    def __call__(self, image, target=None):
        image_size = image.size
        image_size = torch.tensor(image_size[::-1])
        if target is None:
            target = {}

        # out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()
        out_desired_size = torch.tensor([self.desired_size, self.desired_size])

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().int().clamp(min=1)

        scaled_image = F.resize(image, scaled_size.tolist())

        if target is not None:
            target = self.rescale_target(scaled_size, image_size, target)

        # randomly crop or pad images
        delta = scaled_size - out_desired_size
        output_image = scaled_image

        w, h = scaled_image.size
        target["scale"] = [w / self.desired_size, h / self.desired_size]

        if delta.lt(0).any():
            padding = torch.clamp(-delta, min=0)
            if self.random:
                padding1 = (torch.rand(1) * padding).round().int()
                padding2 = padding - padding1
                padding = padding1.tolist()[::-1] + padding2.tolist()[::-1]
            else:
                padding = [0, 0] + padding.tolist()[::-1]
            output_image = F.pad(output_image, padding, 255)
            # output_image = F.pad(scaled_image, [0, 0, padding[1].item(), padding[0].item()])
            if target is not None:
                target = self.pad_target(padding, target)

        if delta.gt(0).any():
            # Selects non-zero random offset (x, y) if scaled image is larger than desired_size.
            max_offset = torch.clamp(delta, min=0)
            if self.random:
                offset = (max_offset * torch.rand(2)).floor().int()
            else:
                offset = torch.zeros(2)
            region = (offset[0].item(), offset[1].item(), out_desired_size[0].item(), out_desired_size[1].item())
            output_image = F.crop(output_image, *region)
            if target is not None:
                target = self.crop_target(region, target)

        return output_image, target


class RandomDistortion(object):
    """
    Distort image w.r.t hue, saturation and exposure.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, prob=0.5):
        self.prob = prob
        self.tfm = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, img, target=None):
        if np.random.random() < self.prob:
            return self.tfm(img), target
        else:
            return img, target
