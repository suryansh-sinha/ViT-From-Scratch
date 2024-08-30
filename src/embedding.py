import torch
import torch.nn as nn

class Embeddings(nn.Module):
    """
    We first convert the image into patch embeddings using a convolution function.
    Each patch is considered as a token. The embedding size is 768.
    If we input an image of 224x224, we assign patch size as 16x16.
    Then we have 14 patches in each row and column.
    """
    def __init__(self, img_size, patch_size, channels=3, emb_dim=768):
        super().__init__()
        self.img_s = img_size
        self.patch_s = patch_size
        self.chn = channels
        self.emb = emb_dim
        # Calculate the number of patches from the image size and patch size
        self.npatches = (img_size // patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size embedding_size
        self.project = nn.Conv2d(channels, emb_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x -> (batch_size, num_channels, image_size, image_size)
        x = self.project(x)  # (batch_size, embedding_dim, patch_height, patch_width)
        # Flattening the last 2 dimensions into a single dimension.
        x = x.flatten(2)        # (batch_size, embedding_size, patch_height * patch_width)
        # Swapping dimension 1 and 2.
        x = x.transpose(1, 2)   # (batch_size, num_patches, embedding_size)
        return x
    
# test = torch.randn([1, 3, 224, 224], dtype=torch.float32)
# embed = Embeddings(224, 16, 3)
# test_output = embed(test)