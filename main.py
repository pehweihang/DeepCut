import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Tuple

import hydra
import torch
from hydra.core.config_store import ConfigStore
from sklearn.metrics import jaccard_score
from torch_geometric.data import Data
from tqdm import tqdm

import util
from extractor import ViTExtractor
from features_extract import deep_features

logger = logging.getLogger(__name__)
PRETRAINED_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth"


class Cut(Enum):
    CC = 0
    NCut = 1


class Dataset(Enum):
    DUTS = "./datasets/DUTS/"


@dataclass
class Config:
    cut: Cut
    dataset: Dataset
    alpha: int = 7
    epochs: int = 10
    k: int = 2
    pretrained_weights_path: str = "./pretrained.pth"
    res: Tuple[int, int] = (280, 280)
    stride: int = 4
    facet: str = "key"
    layer: int = 11


cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(config_name="config", version_base="1.2")
def main(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Download pretrained weights if not exist
    if not os.path.exists(cfg.pretrained_weights_path):
        util.download_url(PRETRAINED_URL, cfg.pretrained_weights_path)

    extractor = ViTExtractor(
        "dino_vits8", cfg.stride, model_dir=cfg.pretrained_weights_path, device=device
    )

    if cfg.cut == Cut.CC:
        from gnn_pool_cc import GNNpool
    else:
        from gnn_pool import GNNpool

    model = GNNpool(384, 64, 32, cfg.k, device).to(device)
    torch.save(model.state_dict(), "model.pt")
    model.train()

    miou = 0

    test_dataset = util.create_dataset(os.path.join(cfg.dataset.value, "test"))

    for sample in tqdm(test_dataset):
        im, label = sample["image"], sample["label"]
        image_tensor, image = util.transform_image(im, cfg.res)
        label_tensor, label_image = util.transform_mask(label, cfg.res)

        F = deep_features(
            image_tensor, extractor, cfg.layer, cfg.facet, bin=False, device=device
        )
        W = util.create_adj(F, cfg.cut, cfg.alpha)

        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = util.load_data(W, F)
        data = Data(node_feats, edge_index, edge_weight).to(device)

        # re-init weights and optimizer for every image
        model.load_state_dict(
            torch.load("./model.pt", map_location=torch.device(device))
        )
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)

        for _ in range(cfg.epochs):
            opt.zero_grad()
            A, S = model(data, torch.from_numpy(W).to(device))
            loss = model.loss(A, S)
            loss.backward()
            opt.step()

        S = S.detach().cpu()
        S = torch.argmax(S, dim=-1)
        mask, S = util.graph_to_mask(S, True, cfg.stride, image_tensor, image)
        sample_miou = jaccard_score(mask.flatten(), (label_image > 122).flatten())
        # util.save_or_show(
        #     [image, mask, util.apply_seg_map(image, mask, alpha=0.7), label_image],
        #     filename="",
        #     dir="",
        #     save=False,
        # )
        miou += sample_miou
    logger.info(f"MIOU: {miou / len(test_dataset)}")


if __name__ == "__main__":
    main()
