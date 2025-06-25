import torch
import torch.nn as nn
import torch.nn.functional as F

class JetBlock(nn.Module):
    """
    The custom residual block for the JetNet model.
    It includes two 3x3 convolutions with a skip connection.
    The first convolution can be dilated.
    """
    def __init__(self, in_channels, out_channels, dilation=1):
        super(JetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        # Store the input for the residual connection
        residual = x
        
        # First convolutional layer
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second convolutional layer
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add the residual connection
        out += residual
        
        # Final activation
        out = self.relu(out)
        return out

class JetNet(nn.Module):
    """
    The main JetNet model for semantic segmentation.
    It's an encoder-style network with a final classifier and upsampling.
    """
    def __init__(self, num_classes=21):
        super(JetNet, self).__init__()
        # Initial downsampling layer
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        # A sequence of JetBlocks without dilation
        self.layer2 = nn.Sequential(
            JetBlock(32, 32),
            JetBlock(32, 32)
        )
        
        # Downsampling followed by JetBlocks with dilation=2
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            JetBlock(64, 64, dilation=2),
            JetBlock(64, 64, dilation=2)
        )
        
        # Downsampling followed by JetBlocks with dilation=4
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            JetBlock(128, 128, dilation=4),
            JetBlock(128, 128, dilation=4)
        )
        
        # Final 1x1 convolution to produce class scores
        self.classifier = nn.Conv2d(128, num_classes, kernel_size=1)

    def forward(self, x):
        # Pass input through the encoder layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Get class scores from the classifier
        x = self.classifier(x)
        
        # Upsample the output to the original size (total stride is 8)
        x = F.interpolate(x, scale_factor=8, mode='bilinear', align_corners=False)
        return x

if __name__ == '__main__':
    # Example usage:
    # Create a dummy input tensor (batch_size=1, channels=3, height=224, width=224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Initialize the model with 21 classes (e.g., Pascal VOC)
    model = JetNet(num_classes=21)
    
    # Get the model output
    output = model(dummy_input)
    
    # Print the shapes to verify
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

