import torch
import torch.nn as nn
import torchvision

class VGGFeatureExtractor(nn.Module):
    def __init__(self, vgg_model):
        super(VGGFeatureExtractor, self).__init__()
        self.features = vgg_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Linear(512 * 7 * 7, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        output = self.classifier(features)
        return features, output

class FHEFriendlyMLPClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(FHEFriendlyMLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.poly_activation(x)
        x = self.fc2(x)
        x = self.poly_activation(x)
        x = self.fc3(x)
        return x

    def poly_activation(self, x):
        return 0.5 * x + 0.25 * x**2

def count_flops(model, input_size):
    def hook_fn(module, input, output):
        nonlocal flops
        if isinstance(module, nn.Conv2d):
            flops += 2 * input[0].size(1) * output.size(1) * output.size(2) * output.size(3) * module.kernel_size[0] * module.kernel_size[1] // (module.stride[0] * module.stride[1])
        elif isinstance(module, nn.Linear):
            flops += 2 * input[0].size(1) * output.size(1)
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            flops += input[0].size(1) * input[0].size(2) * input[0].size(3)

    flops = 0
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.AdaptiveAvgPool2d)):
            hooks.append(module.register_forward_hook(hook_fn))

    input = torch.randn(input_size)
    model(input)

    for hook in hooks:
        hook.remove()

    # Count FLOPs for poly_activation
    if isinstance(model, FHEFriendlyMLPClassifier):
        flops += 3 * input.numel() * 2  # 3 ops per element, 2 activations

    return flops

def main():
    # VGG Feature Extractor
    vgg_model = torchvision.models.vgg16(weights=None)
    vgg_feature_extractor = VGGFeatureExtractor(vgg_model)
    vgg_feature_extractor.eval()
    vgg_input = torch.randn(1, 3, 32, 32)
    vgg_flops = count_flops(vgg_feature_extractor, vgg_input.shape)
    print(f"VGG Feature Extractor FLOPs: {vgg_flops:,}")

    # MLP Classifier
    mlp = FHEFriendlyMLPClassifier(512 * 7 * 7, 10)
    mlp.eval()
    mlp_input = torch.randn(1, 512 * 7 * 7)
    mlp_flops = count_flops(mlp, mlp_input.shape)
    print(f"MLP Classifier FLOPs: {mlp_flops:,}")

    # Total FLOPs
    total_flops = vgg_flops + mlp_flops
    print(f"Total FLOPs (VGG + MLP): {total_flops:,}")

if __name__ == "__main__":
    main()
