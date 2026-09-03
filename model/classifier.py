from torch import nn
import torch.nn.functional as F

from model.backbone.densenet import densenet121
from model.global_pool import GlobalPool
from model.attention_map import AttentionMap


class Classifier(nn.Module):

    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.cfg = cfg
        self.backbone = densenet121(cfg)

        self.global_pool = GlobalPool(cfg)

        self.expand = 1
        if cfg.global_pool == 'AVG_MAX':
            self.expand = 2
        elif cfg.global_pool == 'AVG_MAX_LSE':
            self.expand = 3

        self._init_classifier()
        self._init_bn()
        self._init_attention_map()

    def _init_classifier(self):
        in_channels = self.backbone.num_features * self.expand

        for index, num_class in enumerate(self.cfg.num_classes):
            fc = nn.Conv2d(
                in_channels,
                num_class,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=True
            )

            fc.weight.data.normal_(0, 0.01)
            fc.bias.data.zero_()

            setattr(self, f"fc_{index}", fc)

    def _init_bn(self):
        in_channels = self.backbone.num_features * self.expand

        for index, _ in enumerate(self.cfg.num_classes):
            bn = nn.BatchNorm2d(in_channels)
            setattr(self, f"bn_{index}", bn)

    def _init_attention_map(self):
        self.attention_map = AttentionMap(
            self.cfg,
            self.backbone.num_features
        )

    def forward(self, x):
        feat_map = self.backbone(x)

        logits = []
        logit_maps = []

        for index, _ in enumerate(self.cfg.num_classes):
            if self.cfg.attention_map != "None":
                feat_map = self.attention_map(feat_map)

            classifier = getattr(self, f"fc_{index}")

            logit_map = None
            if self.cfg.global_pool not in ['AVG_MAX', 'AVG_MAX_LSE']:
                logit_map = classifier(feat_map)
                logit_maps.append(logit_map.squeeze())

            feat = self.global_pool(feat_map, logit_map)

            if self.cfg.fc_bn:
                bn = getattr(self, f"bn_{index}")
                feat = bn(feat)

            feat = F.dropout(feat, p=self.cfg.fc_drop, training=self.training)

            logit = classifier(feat)
            logit = logit.squeeze(-1).squeeze(-1)

            logits.append(logit)

        return logits, logit_maps