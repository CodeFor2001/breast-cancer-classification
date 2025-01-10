import streamlit as st
import torch
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import numpy as np
import torchvision.models as models
import torch.nn as nn

st.set_page_config(
    page_title="Classification",
)

class_names = ['adenosis', 'ductal_carcinoma', 'fibroadenoma', 'lobular_carcinoma', 'mucinous_carcinoma', 'papillary_carcinoma', 'phyllodes_tumor', 'tubular_adenoma']
    

class PatchEmbed(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = torch.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = torch.nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class SwinTransformerBlock(torch.nn.Module):
    def __init__(self, dim, num_heads, window_size=7, mlp_ratio=4., qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        
        self.norm1 = torch.nn.LayerNorm(dim)
        self.qkv = torch.nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(dim, dim)
        
        self.norm2 = torch.nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, mlp_hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        B, L, C = x.shape
        
        # Self attention
        shortcut = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        x = self.proj(x)
        x = shortcut + x
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class SwinTransformer(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=8, 
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24]):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.depths = depths
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, 
                                    in_chans=in_chans, embed_dim=embed_dim)
        
        # Transformer layers
        self.layers = torch.nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = torch.nn.ModuleList([
                SwinTransformerBlock(
                    dim=embed_dim * 2**i_layer,
                    num_heads=num_heads[i_layer],
                    window_size=7
                )
                for _ in range(depths[i_layer])
            ])
            self.layers.append(layer)
            
            if i_layer < self.num_layers - 1:
                # Add patch merging layer
                self.layers.append(
                    torch.nn.Sequential(
                        torch.nn.LayerNorm(embed_dim * 2**i_layer),
                        torch.nn.Linear(embed_dim * 2**i_layer, embed_dim * 2**(i_layer + 1))
                    )
                )
        
        self.norm = torch.nn.LayerNorm(embed_dim * 2**(self.num_layers-1))
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.head = torch.nn.Linear(embed_dim * 2**(self.num_layers-1), num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        
        for i_layer in range(self.num_layers):
            # Transformer blocks
            for block in self.layers[i_layer * 2]:
                x = block(x)
            
            # Patch merging
            if i_layer < self.num_layers - 1:
                x = self.layers[i_layer * 2 + 1](x)
        
        x = self.norm(x)
        x = self.avgpool(x.transpose(1, 2)).squeeze(2)
        x = self.head(x)
        return x

class VisionTransformer(torch.nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=8,
                 embed_dim=384, depth=8, n_heads=8, mlp_ratio=4, qkv_bias=True, drop_rate=0.15):
        super().__init__()
        self.patch_embed = torch.nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = torch.nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.pos_drop = torch.nn.Dropout(p=drop_rate)

        self.blocks = torch.nn.ModuleList([
            torch.nn.ModuleDict({
                "norm1": torch.nn.LayerNorm(embed_dim),
                "attention": torch.nn.MultiheadAttention(embed_dim, n_heads, dropout=drop_rate, batch_first=True),
                "norm2": torch.nn.LayerNorm(embed_dim),
                "mlp": torch.nn.Sequential(
                    torch.nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
                    torch.nn.GELU(),
                    torch.nn.Dropout(drop_rate),
                    torch.nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
                    torch.nn.Dropout(drop_rate)
                )
            })
            for _ in range(depth)
        ])

        self.norm = torch.nn.LayerNorm(embed_dim)
        self.head = torch.nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)  # (B, N, E)
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, E)
        x = torch.cat((cls_token, x), dim=1)  # (B, N+1, E)
        x = x + self.pos_embed  # Add positional embedding
        x = self.pos_drop(x)

        for block in self.blocks:
            x_norm1 = block["norm1"](x)
            # Properly pass query, key, and value to attention
            attn_output, _ = block["attention"](x_norm1, x_norm1, x_norm1)
            x = x + attn_output  # Residual connection

            x_norm2 = block["norm2"](x)
            mlp_output = block["mlp"](x_norm2)
            x = x + mlp_output  # Residual connection

        x = self.norm(x)
        return self.head(x[:, 0])

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=8):
        super(CustomAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),  # features.0
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),  # features.3
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),  # features.6
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),  # features.8
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),  # features.10
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),  # classifier.1
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),  # classifier.4
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),  # classifier.6
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x


@st.cache_resource
def load_swin_model():
    model = SwinTransformer()
    try:
        checkpoint = torch.load('models/best_swin_model.pth', map_location=torch.device('cpu'))    
        # Load and print a few sample weights
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    except Exception as e:
        st.error(f"Error loading checkpoint: {str(e)}")
        raise e
    
    model.eval()
    return model

@st.cache_resource
def load_vit_model():
    # Initialize the model architecture
    model = VisionTransformer(
        img_size=224, patch_size=16, in_channels=3, num_classes=8,
        embed_dim=384, depth=8, n_heads=8, mlp_ratio=4, qkv_bias=True, drop_rate=0.15
    )

    # Load the state dictionary
    checkpoint = torch.load('models/best_vit_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model

def load_cnn_model():
    # Create the AlexNet-like model
    model = CustomAlexNet(num_classes=8)
    
    # Load the state dictionary
    state_dict = torch.load('models/best_model.pth', map_location=torch.device('cpu'))
    
    # Load the state dictionary into the model
    model.load_state_dict(state_dict)
    
    # Set to evaluation mode
    model.eval()

    return model


def transform_vit_image(image):
    transform = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=(-10, 10)),
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

def transform_image(image):
    """Updated transform pipeline with less augmentation during inference"""
    transform = v2.Compose([
        v2.Resize(size=(224, 224)),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def main():
    
    st.sidebar.image("uos.png", use_container_width=True, caption="Breast Cancer Detection")


    st.title("Breast Cancer Classification")

    model_type = st.pills("Select the model to use:", ("Vision Transformer (ViT)", "Swin Transformer", "Convolutional Neural Network (CNN)"), default="Swin Transformer")
    
    uploaded_file = st.file_uploader("Upload an image to predict the type of breast tumor.", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        col1, col2 = st.columns(2)
        # Show original image
        with col1:
            image = Image.open(uploaded_file).convert("RGB")
            st.write("Original Image:")
            st.image(image, caption="Original", use_container_width=True)
     

        # Show preprocessed image
        input_tensor = transform_image(image)
        preprocessed_img = input_tensor.squeeze(0).permute(1, 2, 0).numpy()
        # Denormalize
        preprocessed_img = preprocessed_img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        preprocessed_img = np.clip(preprocessed_img, 0, 1)
        
        with col2:
            with st.spinner('Analyzing...'):
                if model_type == "Vision Transformer (ViT)":
                    input_tensor = transform_vit_image(image)
                    model = load_vit_model()
                elif model_type == "Swin Transformer":
                    input_tensor = transform_image(image)
                    model = load_swin_model()
                else:
                    input_tensor = transform_image(image)
                    model = load_cnn_model()
                
            with torch.no_grad():
                # Get raw outputs
                output = model(input_tensor)
                # Apply temperature scaling to sharpen predictions
                temperature = 0.5  # Adjust this value if needed
                scaled_output = output[0] / temperature
                
                # Calculate probabilities
                probabilities = torch.nn.functional.softmax(scaled_output, dim=0)
                
                # Sort probabilities in descending order
                sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
                
                # Display top 3 predictions

                st.write("### Top 3 Predictions:")
                for i in range(3):
                    class_name = class_names[sorted_indices[i]]
                    prob = sorted_probs[i].item()
                    st.write(f"{i+1}. {class_name}: {prob*100:.2f}%")

            # # Display full distribution
            # st.write("\n### Full Probability Distribution:")
            # prob_dict = {class_names[i]: f"{prob:.4f}" for i, prob in enumerate(probabilities.tolist())}
            # st.write(prob_dict)

if __name__ == "__main__":
    main()
