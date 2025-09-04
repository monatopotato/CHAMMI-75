import torch
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Dataset
from typing import Any, Optional
import pandas as pd
import zipfile
from torchvision.io import decode_image
import cv2
import numpy as np
import random
import json

from dataset.guided_crop import GuidedCrop


class CHAMMIv2(Dataset):
    POS_PREFIX = "pos_"

    def __init__(
        self,
        data_size: str,
        metadata_path: str,
        image_zip_path: str,
        sample_pair_path: str,
        sample_pair: str,
        split: str,
        transform_single_channel_after_resize: Any = None,
        transform_image: Any = None,
        channel_return_numpy: bool = False,
        metadata_type: str = "multichannel",
        use_guided_crops: bool = False,
        guided_crops_path: Optional[str] = None,
        image_size: Optional[tuple[int, int]] = None,
    ):
        """
        Args:
            data_size (str): Size of the dataset. One of {small, medium, large}
            metadata_path (str): Path to the metadata file (a csv file)
            image_zip_path (str): Path to the zip file containing images.
            sample_pair (str): choices = {"simclr", "supcon", None}, whether to sample another view for simclr, another image from the same category for supervised contrastive learning, or None for normal training
            sample_pair_path (str): Path to the json file, {"category_id": [image_index1, image_index2, ...]} for sampling positive images
            split (str): The split of the dataset to use. One of {full, cate_atleast_100samples, cate_atleast_10samples}
            transform_multi_channel (callable, optional): A function/transform that takes in a multi-channel image
                and returns a transformed version.
            channel_return_numpy (bool): Whether to load single-channel images as numpy arrays or PyTorch tensors.
            metadata_type (str): choices = {multichannel, singlechannel}, train single-channel images or multi-channel images
        """
        metadata_path = metadata_path.format(DATASIZE_PLACEHODER=data_size, METADATA_TYPE_PLACEHODER=metadata_type)
        image_zip_path = image_zip_path.format(DATASIZE_PLACEHODER=data_size)
        sample_pair_path = sample_pair_path.format(DATASIZE_PLACEHODER=data_size, SPLIT_PLACEHODER=split)
        self.transform_single_channel_after_resize = transform_single_channel_after_resize
        self.transform_image = transform_image
        self.image_zip_path = image_zip_path
        self.channel_return_numpy = channel_return_numpy
        self.sample_pair = sample_pair
        self.image_size = image_size
        self.use_guided_crops = use_guided_crops
        self.guided_crops_path = guided_crops_path

        assert sample_pair in [None, "simclr", "supcon"], f"Unknown sample_pair: {sample_pair}. Choose one of {{None, 'simclr', 'supcon'}}."

        ## Load metadata file
        if metadata_type == "multichannel":
            ## single_channel_paths and channel_ids are stored as list (some channels) in the csv file
            metadata = pd.read_csv(metadata_path, converters={"single_channel_paths": eval, "channel_ids": eval})
        else:
            ## single_channel_paths and channel_ids are stored as str (only 1 channel) in the csv file
            metadata = pd.read_csv(metadata_path)

        if self.sample_pair == "supcon":  ## for supervised contrastive learning
            ## We load a pre-computed json file for sampling positive images
            ## json file: {"category_id": [image_index1, image_index2, ...]}
            ## We're gonna use the image_index here to get the corresponding row in the metadata
            with open(sample_pair_path, "r") as f:
                cate_to_image_index = json.load(f)
            self.cate_to_image_index = cate_to_image_index

        ## filter metadata based on the split
        if split == "full":
            self.label_col = "label_id"
        else:
            self.label_col = f"{split}_label_id"
            metadata = metadata[metadata[split] == True].reset_index(drop=True)
            ## make sure to `reset_index` for supervised contrastive learning's sampling

        ### only keep necessary columns
        self.metadata = metadata[["single_channel_paths", "channel_ids", self.label_col]]

        ## initialize zip file
        self.images_zip = None

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        single_images = self._get_single_image_list(idx)

        data = self._get_item_helper(idx, single_images=single_images)

        ##### Supervised Contrastive Learning: sample another image from the same category
        if self.sample_pair == "supcon":  ## for supervised contrastive learning to train multichannel images
            ## we sample another index from the same category
            label = str(self.metadata.iloc[idx][self.label_col])  ## current category
            all_positive_imgs = self.cate_to_image_index[label]  ## all image indices in the same category
            N = len(all_positive_imgs)
            if N == 1:  ## sample the same image, which may be slightly different due to augmentation
                data_positive = self._get_item_helper(idx, prefix=self.POS_PREFIX, single_images=single_images)
            else:  ## sample a different image
                ## sample an local position for the first N-1 positions
                local_position = random.randint(0, N - 2)
                pos_idx = all_positive_imgs[local_position]  # get the corresponding index in the metadata
                if pos_idx == idx:  ## if the sample index is the same as the current index
                    pos_idx = all_positive_imgs[N - 1]  ## get the last index
                data_positive = self._get_item_helper(pos_idx, prefix=self.POS_PREFIX)  ## Note: single_images=None here
            assert data["label"] == data_positive["label"], "The label of the positive sample is not the same as the current sample!"
            data.update(data_positive)

        ### for SimCLR: sample another view of the same image
        elif self.sample_pair == "simclr":
            data_positive = self._get_item_helper(idx, prefix=self.POS_PREFIX, single_images=single_images)
            data.update(data_positive)

        return data

    def _get_single_image_list(self, idx: int) -> list[torch.Tensor]:
        row = self.metadata.iloc[idx]
        single_channel_paths = self._ensure_list(row["single_channel_paths"])

        single_images = self.load_image(single_channel_paths)
        return single_images  ## list of tensors, type uint8

    def _get_item_helper(self, idx: int, prefix: str = "", single_images: list[torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        row = self.metadata.iloc[idx]
        single_channel_paths = self._ensure_list(row["single_channel_paths"])
        channel_ids = self._ensure_list(row["channel_ids"])
        label = row[self.label_col]

        if single_images is None:
            single_images = self.load_image(single_channel_paths)

        ################## Resize each single-channel image ###############
        ###################################################################
        if self.guided_crops is not None:
            single_images = [self.resize(self.guided_crops(img, path)) for img, path in zip(single_images, single_channel_paths)]
        else:
            single_images = [self.resize(img) for img in single_images]

        ################# Transform each single-channel image after resizing ##################
        ## Some augmentations can only be applied to single-channel or 3-channel images, e.g., torchvision's ColorJitter
        if self.transform_single_channel_after_resize:
            single_images = [self.transform_single_channel_after_resize(img) for img in single_images]

        ################# Stack single-channel images into a multi-channel image ##################
        if self.channel_return_numpy:
            multi_channel_image = np.stack(single_images, axis=0)
            multi_channel_image = np.transpose(multi_channel_image, (1, 2, 0))  ## to HWC
        else:
            multi_channel_image = torch.concat(single_images, dim=0)

        ################# Transform the whole image ##################
        if self.transform_image:
            multi_channel_image = self.transform_image(multi_channel_image)

        data = {f"{prefix}image": multi_channel_image, f"{prefix}channel_ids": channel_ids, "label": label}
        return data

    def _ensure_list(self, item: Any) -> list:
        if isinstance(item, list):
            return item
        else:
            return [item]

    def load_image(self, single_channel_paths: list[str] | str) -> list[torch.Tensor]:
        images = []
        zf = self.images_zip

        for path in single_channel_paths:
            data: bytes = zf.read(path)  # bytearray(zf.read(path))

            if self.channel_return_numpy:
                # decode via OpenCV directly into NumPy
                arr = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                # convert to float32 in-place
                # img = img.astype(np.float32, copy=False)
                images.append(img)
            else:
                # using Torch's built-in decoder:
                buf = torch.frombuffer(data, dtype=torch.uint8)
                img_t = decode_image(buf)  # .to(torch.float32)
                images.append(img_t)
        return images  ## list of tensors, type uint8


def chammiv2_collate_fn(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor | list[list[int]] | int]:
    POS_PREFIX = CHAMMIv2.POS_PREFIX
    channel_ids_list = [item["channel_ids"] for item in batch]
    all_images = [item["image"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    num_images = len(all_images)

    combine_positive_pair = f"{POS_PREFIX}image" in batch[0]
    if combine_positive_pair:
        channel_ids_list_pos_pair = [item[f"{POS_PREFIX}channel_ids"] for item in batch]
        channel_ids_list += channel_ids_list_pos_pair
        all_images += [item[f"{POS_PREFIX}image"] for item in batch]
        labels = torch.cat([labels, labels], dim=0)
        num_images = num_images * 2

    num_channels_list = [len(channels) for channels in channel_ids_list]
    max_num_channels = max(num_channels_list)
    channel_masks = torch.arange(max_num_channels).unsqueeze(0) < torch.tensor(num_channels_list).unsqueeze(1)

    # Initialize a tensor for padding with zeros
    h, w = batch[0]["image"].shape[-2:]
    images = torch.zeros(num_images, max_num_channels, h, w, dtype=torch.float32)

    for i, (channel_ids, image) in enumerate(zip(channel_ids_list, all_images)):
        num_channels = len(channel_ids)
        images[i, :num_channels] = image

    return {
        "image": images,
        "label": labels,
        "max_channel_len": max_num_channels,
        "channel_ids_list": channel_ids_list,
        "channel_mask": channel_masks,
    }


def get_chammiv2_num_classes(metadata_path: str, split: str) -> int:
    metadata = pd.read_csv(metadata_path)
    if split == "full":
        label_col = "label_id"
    else:
        label_col = f"{split}_label_id"
    return metadata[label_col].nunique()


def get_chammiv2_dataloaders(
    data_size: str,
    metadata_path: str,
    image_zip_path: str,
    sample_pair_path: str,
    sample_pair: str,
    split: str,
    image_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
    augmentation: dict[str, Any],
    metadata_type: str = "multichannel",
    use_guided_crops: bool = False,
    guided_crops_path: Optional[str] = None,
    augment_on_gpu: bool = False,
    **kwargs: Any,
) -> tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Only return train_loader for now.
    """
    train_loader, valid_loader, test_loader = None, None, None
    if augmentation["train"] == "simclr":
        kernel_size = 11
        channel_return_numpy = False  ## return single-channel images as PyTorch tensors

        transform_single_channel_after_resize = None
        transform_image = None
        if not augment_on_gpu:
            if metadata_type == "singlechannel":
                transform_image = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))], p=0.5),
                        transforms.Lambda(lambda x: x.float() / 255.0),
                    ]
                )
            else:  ## multichannel
                ## some torchvison augmentations like ColorJitter can only be applied to 1 or 3-channel images, not for various channels
                ## may use others like Kornia or Albumentations later
                transform_single_channel_after_resize = transforms.Compose(
                    [
                        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4)], p=0.8),
                    ]
                )
                transform_image = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        # transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)], p=0.8), ## already applied to each single-channel image
                        transforms.RandomApply([transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2.0))], p=0.5),
                        transforms.Lambda(lambda x: x.float() / 255.0),
                    ]
                )
            print(f"+-+-+-+-+- Using SimCLR augmentation (CPU) for CHAMMIV2 ({metadata_type}) (use_guided_crops={use_guided_crops})....")
    else:
        raise NotImplementedError(f"augmentation {augmentation} is not implemented")

    dataset = CHAMMIv2(
        data_size=data_size,
        metadata_path=metadata_path,
        image_zip_path=image_zip_path,
        sample_pair_path=sample_pair_path,
        sample_pair=sample_pair,
        split=split,
        transform_single_channel_after_resize=transform_single_channel_after_resize,
        transform_image=transform_image,
        channel_return_numpy=channel_return_numpy,
        metadata_type=metadata_type,
        use_guided_crops=use_guided_crops,
        guided_crops_path=guided_crops_path,
        image_size=image_size,
    )
    if sample_pair in ["simclr", "supcon"]:
        # Batch size is halved as we sample another view/positive image to make up for it
        batch_size = batch_size // 2

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=chammiv2_collate_fn,
        persistent_workers=True,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        prefetch_factor=2,
    )
    return train_loader, valid_loader, test_loader


# ============================================
# Worker init function: each worker opens its own ZipFile
# ============================================
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    dataset.images_zip = zipfile.ZipFile(dataset.image_zip_path, "r")

    if dataset.use_guided_crops and dataset.guided_crops_path:
        dataset.guided_crops = GuidedCrop(dataset.guided_crops_path)
        dataset.resize = transforms.Resize(size=dataset.image_size, antialias=True)
    else:
        dataset.guided_crops = None
        dataset.resize = transforms.RandomResizedCrop(size=dataset.image_size, antialias=True)


if __name__ == "__main__":
    ############## Example of using the dataloader ##############cn
    ####  $ cd CHAMMI-75/models
    ####  $ python -m simclr.dataloader.chammiv2
    from torch.utils.data import DataLoader
    import yaml

    ## read data config
    with open("simclr/dataloader/data_config.yaml", "r") as f:
        data_cfg = yaml.safe_load(f)["chammiv2"]

    train_loader, _, _ = get_chammiv2_dataloaders(
        data_size=data_cfg["data_size"],
        metadata_path=data_cfg["metadata_path"],
        image_zip_path=data_cfg["image_zip_path"],
        sample_pair="simclr",
        sample_pair_path=data_cfg["sample_pair_path"],
        split="full",  ## all images in the small dataset
        image_size=(224, 224),
        batch_size=3,
        num_workers=2,
        augmentation=data_cfg["augmentation"],
        metadata_type=data_cfg["metadata_type"],
        use_guided_crops=data_cfg["use_guided_crops"],
        guided_crops_path=data_cfg["guided_crops_path"],
        augment_on_gpu=data_cfg["augment_on_gpu"],
    )

    for i, batch in enumerate(train_loader):
        print(f"\nBatch {i}:")
        print("image shape:", batch["image"].shape)
        print("label shape:", batch["label"].shape)
        print("max_channel_len:", batch["max_channel_len"])
        print("channel_ids_list:", batch["channel_ids_list"])
        print("channel_mask:", batch["channel_mask"].shape)
        if i == 2:
            break

        """Example output:
        Batch 0:
        image shape: torch.Size([6, 1, 224, 224])
        label shape: torch.Size([6])
        max_channel_len: 1
        channel_ids_list: [[17], [13], [5], [17], [13], [5]]
        channel_mask: torch.Size([6, 1])

        Batch 1:
        image shape: torch.Size([6, 1, 224, 224])
        label shape: torch.Size([6])
        max_channel_len: 1
        channel_ids_list: [[16], [12], [22], [16], [12], [22]]
        channel_mask: torch.Size([6, 1])

        Batch 2:
        image shape: torch.Size([6, 1, 224, 224])
        label shape: torch.Size([6])
        max_channel_len: 1
        channel_ids_list: [[16], [22], [18], [16], [22], [18]]
        channel_mask: torch.Size([6, 1])
        """
