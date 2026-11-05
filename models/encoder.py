import torch
import torch.nn as nn
import torchvision
from pathlib import Path


class Res101Encoder(nn.Module):

    def __init__(self, replace_stride_with_dilation=None, pretrained_weights=None):
        super().__init__()

        if isinstance(pretrained_weights, (str, Path)):
            path = Path(pretrained_weights).expanduser()
            if not path.is_file():
                raise FileNotFoundError('Pretrained checkpoint not found: {}. Pass '
                                        'pretrained_weights=/external/weights.pth in the command.'.format(path))
            self.pretrained_weights = torch.load(str(path), map_location='cpu')
        else:
            self.pretrained_weights = pretrained_weights

        _model = torchvision.models.resnet.resnet101(pretrained=False,
                                                     replace_stride_with_dilation=replace_stride_with_dilation)
        self.backbone = nn.ModuleDict()
        for dic, m in _model.named_children():
            self.backbone[dic] = m

        self.reduce1 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
        self.reduce2 = nn.Conv2d(2048, 512, kernel_size=1, bias=False)
        self.reduce1d = nn.Linear(in_features=1000, out_features=1, bias=True)

        self._init_weights()
        self._freeze_bn()

    def _freeze_bn(self):

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):

                m.eval()

                m.weight.requires_grad = False
                m.bias.requires_grad = False

    def train(self, mode=True):

        super().train(mode)

        self._freeze_bn()
        return self

    def forward(self, x):
        features = dict()
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["relu"](x)

        x = self.backbone["maxpool"](x)
        x = self.backbone["layer1"](x)
        x = self.backbone["layer2"](x)
        x = self.backbone["layer3"](x)
        features['down2'] = self.reduce1(x)
        x = self.backbone["layer4"](x)
        features['down3'] = self.reduce2(x)

        t = self.backbone["avgpool"](x)
        t = torch.flatten(t, 1)
        t = self.backbone["fc"](t)
        t = self.reduce1d(t)

        return (features, t)

    def _init_weights(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if self.pretrained_weights is not None:

            if isinstance(self.pretrained_weights, str):
                print(f"Loading weights from: {self.pretrained_weights}")
                raw_state_dict = torch.load(self.pretrained_weights, map_location='cpu')
            else:
                raw_state_dict = self.pretrained_weights

            if 'model' in raw_state_dict:
                print("-> Found 'model' key in pth, extracting state_dict...")
                raw_state_dict = raw_state_dict['model']
            elif 'state_dict' in raw_state_dict:
                raw_state_dict = raw_state_dict['state_dict']

            model_state_dict = self.state_dict()
            new_state_dict = {}

            matched_count = 0
            total_count = len(raw_state_dict)

            print("-> Start matching keys...")

            for k, v in raw_state_dict.items():

                target_key = k

                if k not in model_state_dict and f"backbone.{k}" in model_state_dict:
                    target_key = f"backbone.{k}"

                if target_key in model_state_dict:

                    if v.shape == model_state_dict[target_key].shape:
                        new_state_dict[target_key] = v
                        matched_count += 1
                    else:
                        print(f"!! Shape Mismatch: {k} {v.shape} vs {target_key} {model_state_dict[target_key].shape}")

            if len(new_state_dict) > 0:

                self.load_state_dict(new_state_dict, strict=False)
                print(f"🔥🔥 Success: Loaded {matched_count}/{len(model_state_dict)} layers from pretrained weights.")
            else:
                raise ValueError('No pretrained parameter matched the encoder; refusing silent random initialization')
