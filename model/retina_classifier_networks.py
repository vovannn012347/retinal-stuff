import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights


class FourMetrics(nn.Module):
    def __init__(self, threshold=0.75, correcteness_index=-1):
        super(FourMetrics, self).__init__()
        self.threshold = threshold
        self.correcteness_index = correcteness_index

    '''@staticmethod
    def relative_accuracy(pred, target):
        return 1 - torch.abs(pred - target)  # Relative score [0, 1]'''

    def forward(self, input_sigmoid, target):

        pred = torch.clamp(input_sigmoid, 0, 1)

        pred_binary = (pred >= self.threshold).float()
        target_binary = (target >= self.threshold).float()  # target is already 0 or 1

        # True positives: both prediction and target are 1
        tp = (pred_binary * target_binary).sum().item()

        # False positives: prediction is 1, but target is 0
        fp = ((pred_binary == 1) & (target_binary == 0)).sum().item()

        # False negatives: prediction is 0, but target is 1
        fn = ((pred_binary == 0) & (target_binary == 1)).sum().item()

        # True negatives: both prediction and target are 0
        tn = ((pred_binary == 0) & (target_binary == 0)).sum().item()

        return tp, fp, fn, tn


class IoUWeightedMetrics(nn.Module):
    def __init__(self, weight_positive, weight_negative, reduction='mean'):
        super(IoUWeightedMetrics, self).__init__()
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative
        self.reduction = reduction
        # self.bce_weight = bce_weight

    def forward(self, input_sigmoid, target):

        predicted = torch.clamp(input_sigmoid, 0, 1)
        weights = torch.where(target >= 0.95, self.weight_positive, self.weight_negative)

        intersection = torch.clamp(predicted * target, 0, 1)
        union = torch.clamp(predicted + target, 0, 1)

        intersection_weighted = (intersection * weights).sum(dim=(2, 3), keepdim=True)
        predicted_weighted = (predicted * weights).sum(dim=(2, 3), keepdim=True)
        target_weighted = (target * weights).sum(dim=(2, 3), keepdim=True)
        union_weighted = (union * weights).sum(dim=(2, 3), keepdim=True)

        # Compute IoU for each image in the batch
        iou = (intersection_weighted + 1e-8) / (union_weighted + 1e-8)
        precision = (intersection_weighted + 1e-8) / (predicted_weighted + 1e-8)
        recall = (intersection_weighted + 1e-8) / (target_weighted + 1e-8)
        f1_score = (2 * precision * recall + 1e-8) / (precision + recall + 1e-8)

        true_negative = (1 - predicted) * (1 - target) * weights
        tn_weighted = true_negative.sum(dim=(2, 3), keepdim=True)
        total_weighted = weights.sum(dim=(2, 3), keepdim=True)
        accuracy = (intersection_weighted + tn_weighted + 1e-8) / (total_weighted + 1e-8)

        # Apply reduction
        if self.reduction == 'mean':
            metrics = {
                "IoU": iou.mean().item(),
                "Accuracy": accuracy.mean().item(),
                "Precision": precision.mean().item(),
                "Recall": recall.mean().item(),
                "F1-Score": f1_score.mean().item()
            }
        elif self.reduction == 'sum':
            metrics = {
                "IoU": iou.sum().item(),
                "Accuracy": accuracy.sum().item(),
                "Precision": precision.sum().item(),
                "Recall": recall.sum().item(),
                "F1-Score": f1_score.sum().item()
            }
        elif self.reduction == 'none':
            metrics = {
                "IoU": iou,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score
            }
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Choose 'none', 'mean', or 'sum'.")

        return metrics


class WeightedBCELoss(nn.Module):
    def __init__(self, weight_positive, weight_negative, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.weight_positive = weight_positive
        self.weight_negative = weight_negative
        self.reduction = reduction

    def forward(self, input_sigmoid, target):
        # Calculate binary cross-entropy loss
        loss = - (self.weight_positive * target * torch.log(input_sigmoid + 1e-8) +
                  self.weight_negative * (1 - target) * torch.log(1 - input_sigmoid + 1e-8))

        if self.reduction == 'mean':
            return torch.mean(loss)  # Return scalar mean loss
        elif self.reduction == 'sum':
            return torch.sum(loss)  # Return scalar total loss
        elif self.reduction == 'none':
            return loss  # Return element-wise loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}. Choose 'none', 'mean', or 'sum'.")


# old definitor
class FcnskipNerveDefinitor(nn.Module):
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 base=8):
        super(FcnskipNerveDefinitor, self).__init__()

        # Encoder: Convolution + Max Pooling layers
        self.conv1 = self._conv_block_1(input_channels, base)
        self.conv2 = self._conv_block_1(base, base * 2)
        self.conv3 = self._conv_block_1(base * 2, base * 4)
        # self.conv4 = self._conv_block_1(base * 4, base * 8)

        # Pooling for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.intermidiate = nn.Sequential(
            nn.Conv2d(base*4, base*4, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        self.activation = nn.LeakyReLU(inplace=True)

        # Decoder: Transpose Convolutions for Upsampling
        self.decoder1 = self._upconv_block(base * 4, base * 2)
        self.decoder2 = self._upconv_block(base * 2 + base * 4, base * 2)
        self.decoder3 = self._upconv_block(base * 2 + base * 2, base)

        # Final convolution for classification
        self.final_conv = nn.Conv2d(base, num_classes, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.final_activation = nn.Sigmoid()  # For binary segmentation, use Sigmoid

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)  # [B, 32, 576, 576]
        x1_pooled = self.pool(x1)  # [B, 32, 288, 288]

        x2 = self.conv2(x1_pooled)  # [B, 64, 288, 288]
        x2_pooled = self.pool(x2)  # [B, 64, 144, 144]

        x3 = self.conv3(x2_pooled)  # [B, 256, 144, 144]
        x3_pooled = self.pool(x3)  # [B, 256, 72, 72]

        '''x4 = self.conv4(x3_pooled)  # [B, 512, 72, 72]
        x4_pooled = self.pool(x4)  # [B, 512, 36, 36]'''

        x = self.intermidiate(x3_pooled)
        x = self.upsample(x)

        # Decoder (Upsampling)
        x = self.decoder1(x)  # [B, 256, 72, 72]
        x = self.decoder2(torch.cat([x, x3_pooled], dim=1))  # Add skip connection from x3  [B, 128, 144, 144]
        x = self.decoder3(torch.cat([x, x2_pooled], dim=1))  # Add skip connection from x2  [B, 64, 288, 288]

        # Final 1x1 Convolution to reduce to num_classes output
        x = self.final_conv(x)  # [B, num_classes, 576, 576]
        # Final upsample to the original size
        return self.final_activation(x)

    @staticmethod
    def _conv_block_1(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    @staticmethod
    def _upconv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )


class FcnskipNerveDefinitor2(nn.Module):
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 base=64):
        super(FcnskipNerveDefinitor2, self).__init__()

        self.encoder1 = self._conv_block_2(input_channels, base)
        self.encoder2 = self._conv_block_2(base, base*2)
        self.encoder3 = self._conv_block_3(base*2, base*4)
        self.encoder4 = self._conv_block_3(base*4, base*8)

        self.intermidiate = nn.Sequential(
            nn.Conv2d(base*8, base*8, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        # Decoder
        self.decoder4 = self._upconv_block(base*8 + base*4, base*4)
        self.decoder3 = self._upconv_block(base*4 + base*2, base*2)
        self.decoder2 = self._upconv_block(base*2 + base, base)

        self.decoder_1_final_conv = nn.Sequential(
            nn.Conv2d(base, num_classes, kernel_size=1),
            nn.Sigmoid()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    @staticmethod
    def _conv_block_2(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    @staticmethod
    def _conv_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    @staticmethod
    def _upconv_block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )

    def forward(self, x):

        enc1 = self.encoder1(x)
        pool1 = self.pool(enc1)

        enc2 = self.encoder2(pool1)
        pool2 = self.pool(enc2)

        enc3 = self.encoder3(pool2)
        pool3 = self.pool(enc3)

        enc4 = self.encoder4(pool3)
        pool4 = self.pool(enc4)

        interm = self.intermidiate(pool4)
        # Decoder
        up4 = self.upsample(interm)

        dec4 = self.decoder4(torch.cat([up4, pool3], dim=1))
        dec3 = self.decoder3(torch.cat([dec4, pool2], dim=1))
        dec2 = self.decoder2(torch.cat([dec3, pool1], dim=1))

        return self.decoder_1_final_conv(dec2)

    @staticmethod
    def create_model(num_classes=1, input_channels=3):
        # Load VGG-16 pretrained model
        vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        vgg_features = list(vgg16.features.children())  # Extract VGG-16 feature blocks

        # Create the custom model
        model = FcnskipNerveDefinitor2(num_classes=num_classes, input_channels=input_channels)

        # Map pretrained weights from VGG-16 to the custom model
        vgg_layers = [
            model.encoder1,  # Map first two VGG-16 blocks to encoder1
            model.encoder2,  # Map next two VGG-16 blocks to encoder2
            model.encoder3,  # Map next three VGG-16 blocks to encoder3
            model.encoder4  # Map next three VGG-16 blocks to encoder4
        ]

        start_idx = 0
        for i, encoder in enumerate(vgg_layers):
            num_layers = len(list(encoder.children()))
            for j in range(num_layers):
                if isinstance(encoder[j], nn.Conv2d):
                    encoder[j].weight.data = vgg_features[start_idx].weight.data.clone()
                    encoder[j].bias.data = vgg_features[start_idx].bias.data.clone()
                start_idx += 1
            # include pooling - it is not included in encoder layer for skip connections
            start_idx += 1

        # Return the fully initialized model
        return model


class HandmadeGlaucomaClassifier(nn.Module):
    def __init__(self,
                 num_classes=3,
                 input_channels=3,
                 base=64,
                 input_size=128):
        super(HandmadeGlaucomaClassifier, self).__init__()

        # Load pre-trained VGG16
        #vgg = models.vgg16(pretrained=True)

        # Transfer weights from VGG16 for the first 4 blocks
        self.encoder1 = self._conv_block_2(input_channels, base)
        self.encoder2 = self._conv_block_2(base, base*2)
        self.encoder3 = self._conv_block_3(base*2, base*4)
        self.encoder4 = self._conv_block_3(base*4, base*8)

        # Intermediate layer
        self.intermidiate = nn.Sequential(
            nn.Conv2d(base*8, base*8, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
        )

        # Fully connected layers for 3 output channels
        self.flatten = nn.Flatten()

        lin1 = nn.Linear(base*8*int((input_size/16)**2), base)
        lin2 = nn.Linear(base, num_classes)

        self.fc = nn.Sequential(
            lin1, #nn.Linear(base*8 * 7 * 7, 256),  # Adjust based on output of encoder4
            nn.ReLU(),
            lin2, #nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    @staticmethod
    def _conv_block_2(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _conv_block_3(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    @staticmethod
    def _extract_vgg_weights(vgg, start, end):
        # Extracts a subset of VGG layers' weights as a state_dict
        sublayers = list(vgg.features.children())[start:end]
        state_dict = {f"{i}": layer.state_dict() for i, layer in enumerate(sublayers)}
        return {k: v for d in state_dict.values() for k, v in d.items()}

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = self.pool(enc1)

        enc2 = self.encoder2(pool1)
        pool2 = self.pool(enc2)

        enc3 = self.encoder3(pool2)
        pool3 = self.pool(enc3)

        enc4 = self.encoder4(pool3)
        pool4 = self.pool(enc4)

        # Intermediate processing
        interm = self.intermidiate(pool4)

        # Flatten and FC output
        flattened = self.flatten(interm)
        output = self.fc(flattened)

        return output

    @staticmethod
    def create_model(num_classes=1, input_channels=3, input_size=288):
        # Load VGG-16 pretrained model
        vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        vgg_features = list(vgg16.features.children())  # Extract VGG-16 feature blocks

        # Create the custom model
        model = HandmadeGlaucomaClassifier(num_classes=num_classes, input_channels=input_channels, input_size=input_size)

        # Map pretrained weights from VGG-16 to the custom model
        vgg_layers = [
            model.encoder1,  # Map first two VGG-16 blocks to encoder1
            model.encoder2,  # Map next two VGG-16 blocks to encoder2
            model.encoder3,  # Map next three VGG-16 blocks to encoder3
            model.encoder4  # Map next three VGG-16 blocks to encoder4
        ]

        start_idx = 0
        for i, encoder in enumerate(vgg_layers):
            num_layers = len(list(encoder.children()))
            for j in range(num_layers):
                if isinstance(encoder[j], nn.Conv2d):
                    encoder[j].weight.data = vgg_features[start_idx].weight.data.clone()
                    encoder[j].bias.data = vgg_features[start_idx].bias.data.clone()
                start_idx += 1
            # include pooling - it is not included in encoder layer for skip connections
            start_idx += 1

        # Return the fully initialized model
        return model
