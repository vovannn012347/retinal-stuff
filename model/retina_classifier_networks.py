import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import VGG16_Weights


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

#old definitor
'''class FcnskipNerveDefinitor(nn.Module):
    def __init__(self,
                 num_classes=1,
                 input_channels=3,
                 base=8):
        super(FcnskipNerveDefinitor, self).__init__()

        # Encoder: Convolution + Max Pooling layers
        self.conv1 = nn.Conv2d(input_channels, base, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(base, base * 2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(base * 2, base * 4, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(base * 4, base * 8, kernel_size=3, padding=1)

        # Pooling for downsampling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.activation = nn.LeakyReLU(inplace=True)

        # Decoder: Transpose Convolutions for Upsampling
        self.deconv1 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.deconv3 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)

        # Final convolution for classification
        self.final_conv = nn.Conv2d(base, num_classes, kernel_size=1)
        self.final_activation = nn.Sigmoid()  # For binary segmentation, use Sigmoid

    def forward(self, x):
        # Encoder
        x1 = self.activation(self.conv1(x))  # [B, 32, 576, 576]
        x1_pooled = self.pool(x1)  # [B, 32, 288, 288]

        x2 = self.activation(self.conv2(x1_pooled))  # [B, 64, 288, 288]
        x2_pooled = self.pool(x2)  # [B, 64, 144, 144]

        x3 = self.activation(self.conv3(x2_pooled))  # [B, 256, 144, 144]
        x3_pooled = self.pool(x3)  # [B, 256, 72, 72]

        x4 = self.activation(self.conv4(x3_pooled))  # [B, 512, 72, 72]
        x4_pooled = self.pool(x4)  # [B, 512, 36, 36]

        # Decoder (Upsampling)
        x = self.activation(self.deconv1(x4_pooled))  # [B, 256, 72, 72]
        x = self.activation(self.deconv2(x + x3_pooled))  # Add skip connection from x3  [B, 128, 144, 144]
        x = self.activation(self.deconv3(x + x2_pooled))  # Add skip connection from x2  [B, 64, 288, 288]

        # Final 1x1 Convolution to reduce to num_classes output
        x = self.final_conv(x)  # [B, num_classes, 576, 576]
        # Final upsample to the original size
        return self.final_activation(x)'''


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

        #nn.ConvTranspose2d(base*8, base*8, kernel_size=2, stride=2)

        self.upsample = nn.ConvTranspose2d(base*8, base*8, kernel_size=2, stride=2) #nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

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

    def _conv_block_2(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def _conv_block_3(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True)
        )

    def _upconv_block(self, in_channels, out_channels):
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
                 input_size=288):
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

    def _conv_block_2(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _conv_block_3(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def _extract_vgg_weights(self, vgg, start, end):
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
