import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Tuple

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import call, instantiate
from omegaconf import MISSING
from sklearn.metrics import jaccard_score
from torch_geometric.data import Data

import util
from extractor import ViTExtractor
from features_extract import deep_features

logger = logging.getLogger(__name__)
PRETRAINED_URL = "https://dl.fbaipublicfiles.com/dino/dino_deitsmall8_pretrain/dino_deitsmall8_pretrain_full_checkpoint.pth"


@dataclass
class LossFunc:
    _target_: str


@dataclass
class Cut:
    loss_func: LossFunc
    value: int


@dataclass
class NCut(Cut):
    loss_func: LossFunc = LossFunc("ncut_loss.loss")
    value: int = 0


@dataclass
class CC(Cut):
    loss_func: LossFunc = LossFunc("cc_loss.loss")
    value: int = 1


@dataclass
class GNN:
    _target_: str


@dataclass
class GCN(GNN):
    _target_: str = "gcn_pool.GNNpool"


@dataclass
class GAT(GNN):
    _target_: str = "gcn_pool.GNNpool"


@dataclass
class Dataset:
    path: str


@dataclass
class DUTS(Dataset):
    path: str = "./datasets/DUTS/test"


@dataclass
class ECSSD(Dataset):
    path: str = "./datasets/ECSSD/"


@dataclass
class Config:
    show_img: bool = False
    alpha: int = 7
    epochs: int = 10
    k: int = 2
    pretrained_weights_path: str = "./pretrained.pth"
    res: Tuple[int, int] = (280, 280)
    stride: int = 4
    facet: str = "key"
    layer: int = 11

    dataset: Dataset = MISSING
    gnn: GNN = MISSING
    cut: Cut = MISSING

    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
            {"cut": "???"},
            {"gnn": "???"},
            {"dataset": "???"},
        ]
    )
    hydra: Any = field(
        default_factory=lambda: {
            "run": {
                "dir": "outputs/${hydra.runtime.choices.dataset}/${hydra.runtime.choices.cut}-${hydra.runtime.choices.gnn}-alpha${alpha}-k${k}--${now:%Y-%m-%d_%H-%M-%S}"
            }
        }
    )


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
cs.store(group="dataset", name="DUTS", node=DUTS)
cs.store(group="dataset", name="ECSSD", node=ECSSD)
cs.store(group="cut", name="NCut", node=NCut)
cs.store(group="cut", name="CC", node=CC)
cs.store(group="gnn", name="GCN", node=GCN)
cs.store(group="gnn", name="GAT", node=GAT)


@hydra.main(config_name="config", version_base="1.2")
def main(cfg: Config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)

    # Download pretrained weights if not exist
    if not os.path.exists(cfg.pretrained_weights_path):
        util.download_url(PRETRAINED_URL, cfg.pretrained_weights_path)

    extractor = ViTExtractor(
        "dino_vits8", cfg.stride, model_dir=cfg.pretrained_weights_path, device=device
    )

    model = instantiate(cfg.gnn, 384, 64, 32, cfg.k, device).to(device)
    torch.save(model.state_dict(), "model.pt")
    model.train()

    miou = 0

    dataset = util.create_dataset(cfg.dataset.path)

    for i, sample in enumerate(dataset):
        im, label = sample["image"], sample["label"]
        image_tensor, image = util.transform_image(im, cfg.res)
        label_tensor, label_image = util.transform_mask(label, cfg.res)

        F = deep_features(
            image_tensor, extractor, cfg.layer, cfg.facet, bin=False, device=device
        )
        W = util.create_adj(F, cfg.cut.value, cfg.alpha)

        # Data to pytorch_geometric format
        node_feats, edge_index, edge_weight = util.load_data(W, F)
        data = Data(node_feats, edge_index, edge_weight).to(device)

        # re-init weights and optimizer for every image
        model.load_state_dict(
            torch.load("./model.pt", map_location=torch.device(device))
        )
        opt = torch.optim.AdamW(model.parameters(), lr=0.001)

        A, S = None, None
        for _ in range(cfg.epochs):
            opt.zero_grad()
            A, S = model(data, torch.from_numpy(W).to(device))
            loss = call(cfg.cut.loss_func, A, S, cfg.k)
            loss.backward()
            opt.step()

        S = S.detach().cpu()
        S = torch.argmax(S, dim=-1)
        mask, S = util.graph_to_mask(S, False, cfg.stride, image_tensor, image)
        sample_miou = jaccard_score(mask.flatten(), (label_image > 122).flatten())
        logger.info(f"Image {i} - mIOU: {sample_miou}")
        if cfg.show_img:
            util.save_or_show(
                [image, mask, util.apply_seg_map(image, mask, alpha=0.7), label_image],
                filename="",
                dir="",
                save=False,
            )
        miou += sample_miou
    logger.info(f"MIOU: {miou / len(dataset)}")


if __name__ == "__main__":
    main()
