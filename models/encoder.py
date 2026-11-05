"""
Backbones supported by torchvison.
"""
import torch
import torch.nn as nn
import torchvision
from pathlib import Path


class Res101Encoder(nn.Module):
    """
    Resnet101 backbone from deeplabv3
    modify the 'downsample' component in layer2 and/or layer3 and/or layer4 as the vanilla Resnet
    """

    def __init__(self, replace_stride_with_dilation=None, pretrained_weights=None):
        super().__init__()
        # None explicitly requests random initialization for isolated smoke tests.
        # Production defaults never silently continue without pretrained weights.
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
        # 遍历所有模块
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                # 1. 锁定统计量 (Running Mean/Var)，防止被 Support Set 的 1 张图带偏
                m.eval() 
                # 2. 关掉梯度，防止更新权重 (Affine parameters: gamma/beta)
                m.weight.requires_grad = False
                m.bias.requires_grad = False
                
    def train(self, mode=True):
        """
        重写 train 方法是防止你在外部调用 model.train() 时
        意外把 BN 层又开起来了。这是很多复现失败的隐蔽原因。
        """
        super().train(mode)
        # 再次强制冻结 BN
        self._freeze_bn()
        return self

    def forward(self, x):
        features = dict()
        x = self.backbone["conv1"](x)
        x = self.backbone["bn1"](x)
        x = self.backbone["relu"](x)
        # features['down1'] = x
        x = self.backbone["maxpool"](x)
        x = self.backbone["layer1"](x)
        x = self.backbone["layer2"](x)
        x = self.backbone["layer3"](x)
        features['down2'] = self.reduce1(x)
        x = self.backbone["layer4"](x)
        features['down3'] = self.reduce2(x)

        # feature map -> avgpool -> fc -> single value
        # (avgpool): AdaptiveAvgPool2d(output_size=(1, 1)); (fc): Linear(in_features=2048, out_features=1000, bias=True)
        t = self.backbone["avgpool"](x)
        t = torch.flatten(t, 1)
        t = self.backbone["fc"](t)
        t = self.reduce1d(t)

        return (features, t)

    def _init_weights(self):
        # 1. 先进行基础的 Kaiming 初始化 (兜底)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 2. 开始加载预训练权重
        if self.pretrained_weights is not None:
            # 兼容性处理：如果 pretrained_weights 是字符串（路径），则加载它
            # 如果已经是加载好的 dict（如你在 __init__ 里写的），则直接用
            if isinstance(self.pretrained_weights, str):
                print(f"Loading weights from: {self.pretrained_weights}")
                raw_state_dict = torch.load(self.pretrained_weights, map_location='cpu')
            else:
                raw_state_dict = self.pretrained_weights

            # 【修复1】检查是否被 "model" 或 "state_dict" 包裹
            if 'model' in raw_state_dict:
                print("-> Found 'model' key in pth, extracting state_dict...")
                raw_state_dict = raw_state_dict['model']
            elif 'state_dict' in raw_state_dict:
                raw_state_dict = raw_state_dict['state_dict']

            # 获取当前模型的 state_dict
            model_state_dict = self.state_dict()
            new_state_dict = {}
            
            matched_count = 0
            total_count = len(raw_state_dict)

            print("-> Start matching keys...")
            
            # 【修复2】智能匹配 Key
            for k, v in raw_state_dict.items():
                # 标准化 key: 去掉 deeplab 可能自带的 'backbone.' 前缀，稍后根据需要加上
                # 但你的模型里刚好有一个 self.backbone，所以我们需要保留或者凑出 'backbone.'
                
                target_key = k # 默认尝试直接匹配
                
                # 情况A: 文件里是 'conv1'，模型里是 'backbone.conv1' (常见于 ImageNet 权重)
                if k not in model_state_dict and f"backbone.{k}" in model_state_dict:
                    target_key = f"backbone.{k}"
                
                # 情况B: 文件里是 'backbone.conv1'，模型里也是 'backbone.conv1' (常见于 DeepLab 权重)
                # 这种情况不需要改，target_key = k
                
                # 情况C: 文件里是 'classifier...'，模型里不需要 -> 跳过

                if target_key in model_state_dict:
                    # 检查形状是否匹配
                    if v.shape == model_state_dict[target_key].shape:
                        new_state_dict[target_key] = v
                        matched_count += 1
                    else:
                        print(f"!! Shape Mismatch: {k} {v.shape} vs {target_key} {model_state_dict[target_key].shape}")
                
            # 3. 加载匹配好的权重
            if len(new_state_dict) > 0:
                # strict=False 允许部分权重不匹配 (比如 fc 层)
                self.load_state_dict(new_state_dict, strict=False)
                print(f"🔥🔥 Success: Loaded {matched_count}/{len(model_state_dict)} layers from pretrained weights.")
            else:
                raise ValueError('No pretrained parameter matched the encoder; refusing silent random initialization')
