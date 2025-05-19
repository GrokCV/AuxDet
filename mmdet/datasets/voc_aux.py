import os.path as osp
import xml.etree.ElementTree as ET
from typing import List
import numpy as np
from mmengine.fileio import list_from_file
from mmdet.datasets.xml_style import XMLDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class VSBWILDVOCDetDataset(XMLDataset):
    """Dataset for VSWILD VOC Detection with auxiliary information."""

    METAINFO = {
        "classes": ("Target",),
        "palette": [
            (106, 0, 228),  # RGB color for class visualization
        ],
    }

    def __init__(self, img_subdir="PNGImages", ann_subdir="Annotations", **kwargs):
        """Initialization function for the dataset."""
        super().__init__(img_subdir=img_subdir, ann_subdir=ann_subdir, **kwargs)
        self._metainfo["dataset_type"] = None  # Optional, you can add dataset-specific types

    def load_data_list(self) -> List[dict]:
        """Load annotations from XML files and include auxiliary features.

        Returns:
            List[dict]: Annotation info with auxiliary image-level features.
        """
        # Ensure class information is defined
        assert (
            self._metainfo.get("classes", None) is not None
        ), "classes in XMLDataset cannot be None."
        self.cat2label = {cat: i for i, cat in enumerate(self._metainfo["classes"])}

        data_list = []
        # Read image IDs from the annotation file
        img_ids = list_from_file(self.ann_file, backend_args=self.backend_args)

        for img_id in img_ids:
            # Define file paths for image and XML annotation
            file_name = osp.join(self.sub_data_root, self.img_subdir, f"{img_id}.png")
            xml_path = osp.join(self.sub_data_root, self.ann_subdir, f"{img_id}.xml")

            # Parse XML file to extract image-level and instance-level information
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Extract image-level features (width, height, view)
            width = int(root.find("size").find("width").text)
            height = int(root.find("size").find("height").text)
            view = root.find("view").text if root.find("view") is not None else "Unknown"
            band_type = (
                root.find("band_type").text
                if root.find("band_type") is not None
                else "Unknown"
            )

            # Parse instance-level annotations from the XML file
            instances = self._parse_instance_info(root)

            # Combine image-level and instance-level information
            parsed_data_info = {
                "img_id": img_id,
                "img_path": file_name,
                "width": width,
                "height": height,
                "view": view,
                "band_type": band_type,
                "instances": instances
            }

            data_list.append(parsed_data_info)

        return data_list

    def _parse_instance_info(self, raw_ann_info: ET) -> List[dict]:
        """Parse instance (object) information from an XML file.

        Args:
            raw_ann_info (ElementTree): XML element tree for annotations.

        Returns:
            List[dict]: List of instance-level information.
        """
        instances = []
        for obj in raw_ann_info.findall("object"):
            instance = {}
            name = obj.find("name").text

            # Skip objects that are not in the predefined classes
            if name not in self._metainfo["classes"]:
                continue

            # Parse "difficult" flag
            difficult = obj.find("difficult")
            difficult = 0 if difficult is None else int(difficult.text)

            # Parse bounding box coordinates
            bnd_box = obj.find("bndbox")
            if bnd_box is not None:
                bbox = [
                    int(float(bnd_box.find("xmin").text)),
                    int(float(bnd_box.find("ymin").text)),
                    int(float(bnd_box.find("xmax").text)),
                    int(float(bnd_box.find("ymax").text)),
                ]

                instance["bbox"] = bbox
                instance["bbox_label"] = self.cat2label[name]
                instance["ignore_flag"] = difficult  # 0: normal, 1: ignore

                instances.append(instance)

        return instances
