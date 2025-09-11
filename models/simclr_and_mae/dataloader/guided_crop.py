import os
import zipfile
import torch
import random
from safetensors.torch import load
from torchvision import disable_beta_transforms_warning
from torch import Tensor

disable_beta_transforms_warning()


class GuidedCrop:
    def __init__(self, crop_data: str):
        assert isinstance(crop_data, str) and crop_data.endswith("zip")
        self.crop_data = os.path.abspath(os.path.expanduser(crop_data))
        self.data_paths = None  # We'll initialize per worker

    def _init_zip(self):
        if self.data_paths is None:
            self.data = zipfile.ZipFile(self.crop_data, "r")
            self.data_paths = set([file.filename for file in self.data.filelist if not file.is_dir()])

    def _guided_crop(self, sample: torch.Tensor, sample_path: str, crop_size: tuple[float, float]) -> torch.Tensor:
        with self.data.open(sample_path) as f:
            image_height, image_width = sample.shape[1], sample.shape[2]

            possible_centroids = load(f.read())["data"]
            chosen_centroid = possible_centroids[random.randint(0, possible_centroids.shape[0] - 1), :]
            x, y = chosen_centroid[0], chosen_centroid[1]
            crop_height, crop_width = crop_size[0], crop_size[1]

            # Divide by 2, as we want half the crop size on each size of the center point
            x1, y1, x2, y2 = get_crop_location(crop_height // 2, crop_width // 2, y, x, image_height, image_width)

            cropped_sample = sample[:, y1:y2, x1:x2]
            return cropped_sample

    def __call__(self, image_tensor: Tensor, channel_img_path: str) -> Tensor:
        self._init_zip()  # ensure zip is opened in this worker

        dataset = channel_img_path.split(os.sep)[1]
        guided_crop_size = get_crop_size(dataset)
        if guided_crop_size != (-1, -1):
            crop_height = random.randint(int(guided_crop_size[0] * 0.9), int(guided_crop_size[0] * 1.1))
            crop_width = random.randint(int(guided_crop_size[1] * 0.9), int(guided_crop_size[1] * 1.1))
            guided_crop_size = (crop_height, crop_width)

            safetensors_name = to_guidance_path(channel_img_path)
            if safetensors_name in self.data_paths:
                image_tensor = self._guided_crop(image_tensor, safetensors_name, guided_crop_size)
            # else:
            # print(f"Warning: {safetensors_name} not found in zip {self.crop_data}, skipping guided crop.")
        return image_tensor


def get_crop_location(crop_height: int, crop_width: int, y_center: int, x_center: int, image_height: int, image_width: int):
    # subtraction goes up or left, addition goes down or right
    y1 = y_center - crop_height
    y2 = y_center + crop_height
    x1 = x_center - crop_width
    x2 = x_center + crop_width

    if y1 < 0:
        y2 = y2 - y1  # add into down direction -(-) = +
        y1 = 0
    elif y2 > image_height:
        y1 = y1 - (y2 - image_height)  # Move y1 up the difference
        y2 = image_height

    if x1 < 0:
        x2 = x2 - x1
        x1 = 0
    elif x2 > image_width:
        x1 = x1 - (x2 - image_width)
        x2 = image_width

    return x1, y1, x2, y2


def to_guidance_path(img_path: str) -> str:
    """
    Example:
    img_path: CHAMMI-75_small/idr0088/idr0088-plate_1085A-converted/1013608212_E09_T0001F001L01A01Z01C01_series-0_z-0_t-0_channel-0.png
    return: CHAMMI-75_guidance/idr0088/idr0088-plate_1085A-converted/1013608212_E09_T0001F001L01A01Z01C01_series-0_z-0_t-0_channel-0.safetensors
    """
    parts = img_path.split(os.sep)  # split into path components
    rest = os.sep.join(parts[1:])  # drop the first component
    base, _ = os.path.splitext(rest)  # strip extension
    return os.path.join("CHAMMI-75_guidance", base + ".safetensors")


def get_crop_size(dataset):
    crop_sizes = {
        "wtc0001": [(256, 256), (450, 450)],
        "jump0001": [(112, 112), (450, 450)],
        "hpa0018": [(200, 200), (450, 450)],
        "nidr0031": [(128, 128), (250, 250)],
        "nidr0032": [(92, 92), (350, 350)],
        "idr0002": [(114, 114), (350, 350)],
        "idr0088": [(114, 114), (350, 350)],
        "idr0086": (-1, -1),
        "idr0089": (-1, -1),
        "idr0008": [(224, 224), (512, 512)],
        "idr0001": [(145, 145), (350, 350)],
        "idr0003": [(72, 72), (140, 140)],
        "idr0006": [(150, 150), (300, 300)],
        "idr0005": [(150, 150), (300, 300)],
        "idr0009": [(150, 150), (450, 450)],
        "idr0010": [(128, 128), (300, 300)],
        "idr0011": [(72, 72), (200, 200)],
        "idr0012": [(128, 128), (200, 200)],
        "idr0013": [(48, 48), (200, 200)],
        "idr0017": [(56, 56), (300, 300)],
        "idr0020": [(70, 70), (200, 200)],
        "idr0022": [(120, 120), (600, 600)],
        "idr0028": [(200, 200), (500, 500)],
        "idr0030": [(150, 150), (300, 300)],
        "idr0033": [(150, 150), (350, 350)],
        "idr0035": [(200, 200), (400, 400)],
        "idr0037": [(100, 100), (400, 400)],
        "idr0056": [(75, 75), (300, 300)],
        "idr0069": [(100, 100), (300, 300)],
        "idr0080": [(200, 200), (400, 400)],
        "idr0093": [(100, 100), (400, 400)],
        "idr0094": [(50, 50), (150, 150)],
        "idr0120": [(200, 200), (600, 600)],
        "idr0123": [(200, 200), (400, 400)],
        "idr0128": [(50, 50), (300, 300)],
        "idr0130": [(20, 20), (150, 150)],
        "idr0133": [(200, 200), (400, 400)],
        "idr0140": [(50, 50), (200, 200)],
        "idr0145": [(100, 100), (300, 300)],
        "nidr0001": [(300, 300), (500, 500)],
        "nidr0003": (-1, -1),
        "nidr0004": [(600, 600), (-1, -1)],
        "nidr0005": (-1, -1),
        "nidr0006": [(128, 128), (300, 300)],
        "nidr0008": [(84, 84), (400, 400)],
        "nidr0010": [(64, 64), (250, 250)],
        "nidr0011": [(140, 140), (450, 450)],
        "nidr0012": [(45, 45), (400, 400)],
        "nidr0013": [(92, 92), (350, 350)],
        "nidr0014": [(140, 140), (350, 350)],
        "nidr0015": [(140, 140), (350, 350)],
        "nidr0016": [(140, 140), (250, 250)],
        "nidr0017": [(140, 140), (350, 350)],
        "nidr0018": (-1, -1),
        "nidr0019": [(240, 240), (350, 350)],
        "nidr0020": [(184, 184), (400, 400)],
        "nidr0021": (-1, -1),
        "nidr0022": (-1, -1),
        "nidr0023": [(184, 184), (400, 400)],
        "nidr0024": (-1, -1),
        "nidr0025": [(250, 250), (400, 400)],
        "nidr0027": [(200, 200), (-1, -1)],
        "nidr0028": (-1, -1),
        "nidr0029": (-1, -1),
        "nidr0030": (-1, -1),
        "hpa0023": [(256, 256), (512, 512)],
        "default": (-1, -1),
    }
    sizes = crop_sizes.get(dataset, crop_sizes["default"])
    if sizes != (-1, -1):
        return random.choice(sizes)
    return sizes
