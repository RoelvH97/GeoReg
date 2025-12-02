# import necessary libraries
import numpy as np
import SimpleITK as sitk

from pathlib import Path
from torch.utils.data import Dataset


def sitk_to_numpy(filename):
    image = sitk.ReadImage(filename)
    spacing = image.GetSpacing()
    offset = image.GetOrigin()

    # convert to (z, y, x)
    spacing = np.array(spacing)[::-1]
    offset = np.array(offset)[::-1]

    image = sitk.GetArrayFromImage(image)
    return image, spacing, offset


class ISLES2024Dataset(Dataset):
    def __init__(self, config):
        self.config = config
        self.root = Path(config.root)

        # get ids
        self.id_list = self.root.glob('CTATr/sub-stroke*_0000.nii.gz')
        self.id_list = [p.name.split('_')[0] for p in self.id_list]
        self.id_list.sort()

    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self, idx):
        id_ = self.id_list[idx]

        # get paths
        dsa_path = self.root / 'DSATr' / id_
        dsa_msk_path = self.root / 'MAP_maskTr' / id_
        dsa_metadata_path = self.root / 'DSA_arteriesTr' / id_
        cta_path = self.root / 'CTATr' / f'{id_}_0000.nii.gz'
        cta_msk_path = self.root / 'CTA_skullTr' / f'{id_}.nii.gz'

        return {"DSA": dsa_path, "DSA_mask": dsa_msk_path, "DSA_metadata": dsa_metadata_path,
                "CTA": cta_path, "CTA_mask": cta_msk_path}
