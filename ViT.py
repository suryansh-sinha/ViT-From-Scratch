# Training to be done on CIFAR-10 dataset. Contains 3x32x32 images.
# Here we use 4x4 patches. And there are a total of 8*8 = 64 patches.
# Thus number of tokens in embedding = 64 + 1 (for class token)
import torch
import torch.nn as nn
import math

class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.embedding_size = config["embedding_size"]
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size embedding_size
        self.projection = nn.Conv2d(self.num_channels, self.embedding_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, embedding_size)
        # x -> (batch_size, num_channels, image_size, image_size)
        x = self.projection(x)  # (batch_size, embedding_size, num_patches_height, num_patches_width)
        # Flattening the last 2 dimensions into a single dimension.
        x = x.flatten(2)        # (batch_size, embedding_size, num_patches_height * num_patches_width)  
        # Swapping dimension 1 and 2.
        x = x.transpose(1, 2)   # (batch_size, num_patches, embedding_size)
        return x
    
# Converting to final embedding form
class Embeddings(nn.Module):
    """
    Combine patch embeddings with class token. Then add positional embeddings to it.
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.patch_embeddings = PatchEmbeddings(config)
        
        # Creating a learnable CLS token.
        self.cls_token = nn.Parameter(torch.randn(1, 1, config["embedding_size"]), requires_grad=True)
        
        # Creating position embeddings for cls token and patches
        # Sinusoidal functions are not used here. They were used in the original "Attention is all you Need" paper.
        self.position_embeddings = nn.Parameter(torch.randn(1, self.patch_embeddings.num_patches + 1, config["embedding_size"]),
                                                            requires_grad=True)
        
        # Adding dropout
        self.dropout = nn.Dropout(p=config["hidden_dropout_prob"])
        
    def forward(self, x):
        x = self.patch_embeddings(x)    # (batch_size, num_patches, embedding_size)
        batch_size, _, _ = x.size()
        # Expand the cls token to batch size
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, embedding_size)
        # Concatenate cls token with patch embedding
        x = torch.concat((cls_tokens, x), dim=1)    # (batch_size, num_patches + 1, embedding_size)
        # Add positional embeddings to embeddings
        x = x + self.position_embeddings
        x = self.dropout(x)
        return x    # (batch_size, num_patches + 1, embedding_size)
    
    
# Implementing a head of a multi-head attention module.
class AttentionHead(nn.Module):
    """
    Implementing a head of a multi-head attention module.
    Takes sequence of embeddings as inputs (positional included) and computes the query, key, value for each embedding.
    These are then used to calculate the attention weights for each token.
    The attention weights are used to calculate new embeddings using a weighted sum of value vectors.
    """
    def __init__(self, embedding_size, attention_head_size, dropout, bias=True):
        super().__init__()
        self.embedding_size = embedding_size
        self.attention_head_size = attention_head_size  # dimension of the Q, K, V tensors used for Attention.
        
        # Creating the query, key and value projection layers.
        # These are W^Q, W^K, W^V
        self.query = nn.Linear(embedding_size, attention_head_size, bias=bias)
        self.key = nn.Linear(embedding_size, attention_head_size, bias=bias)
        self.value = nn.Linear(embedding_size, attention_head_size, bias=bias)
        
        # For regularization, so that more generalized representation of attention is learned
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # These are the actual Q, K, V
        # [batch_size, num_patches + 1, embedding_size] --> [batch_size, num_patches+1, attention_head_size]
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        
        # Calculate the attention scores using softmax((Q.K^T)/sqrt(d_k)).V
        # key.transpose(1,2) --> [batch_size, attention_head_size, num_patches + 1]
        attention_scores = torch.matmul(query, key.transpose(1, 2)) # [batch_size, num_patches + 1, num_patches + 1]
        attention_scores /= math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=1)
        attention_probs = self.dropout(attention_probs)
        
        # Calculate the attention output
        attention_output = torch.matmul(attention_probs, value) # [batch_size, num_patches + 1, attention_head_size]

        return (attention_output, attention_probs)
        
class MultiHeadAttention(nn.Module):
    """
    Outputs of all the attention heads are concatenated and linearly projected (W^O)
    to obtain the final output of the multi-head attention module.
    """
    
    def __init__(self, config):
        super().__init__()
        self.embedding_size = config["embedding_size"]
        self.num_attention_heads = config["num_attention_heads"]
        # If embedding_size is 512, num_attention_heads = 8, then the embedding is split into 512 / 8.
        # Thus each head gets a dimensional representation of 64.
        # attention_head_size represents the dimensionality of the key, query, and value vectors for each attention head.
        self.attention_head_size = self.embedding_size // self.num_attention_heads
        # This represents the combined dimension after concatenating the outputs of all attention heads.
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # Whether we want to use the bias in Q, K, V.
        self.qkv_bias = config["qkv_bias"]
        
        # Creating a list of attention heads -
        self.heads = nn.ModuleList([])  # Acts the same as a list but it's for torch modules.
        # Creating required number of attention heads and storing them in the list.
        for _ in range(self.num_attention_heads):
            head = AttentionHead(
                self.embedding_size,
                self.attention_head_size,
                config["attention_probs_dropout_prob"],
                self.qkv_bias
            )
            self.heads.append(head)
        
        # Create a linear layer to project the output back to embedding size
        self.output_projection = nn.Linear(self.all_head_size, self.embedding_size)
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        
    def forward(self, x, output_attentions=False):
        attention_outputs = [head(x) for head in self.heads]
        # Getting the attention output from each index in the list, then concatenating them along the last dimension.
        # Shape of attention output is -> [batch_size, num_patches, all_head_sizes]
        attention_output = torch.cat([attention_output for attention_output, _ in attention_outputs], dim=-1)
        attention_output = self.output_projection(attention_output)
        attention_output = self.dropout(attention_output)
        
        if not output_attentions:
            return (attention_output, None)
        else:
            # Each attention prob has shape --> [batch_size, n_patches + 1, n_patches + 1]
            attention_probs = torch.stack([attention_probs for _, attention_probs in attention_outputs], dim=1)
            # This stacking results in a new dimension being added at position 1, and then stacking tensors along that dim.
            # Thus, shape becomes --> [batch_size, num_heads, n_patches + 1, n_patches + 1]
            return (attention_output, attention_probs)
        
class MLP(nn.Module):
    """
    Creating a Multi-Layer Perceptron.
    It has 2 fully connected layers, uses GELU as activation.
    """
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config["embedding_size"], config["intermediate_size"])
        self.gelu = nn.GELU(approximate='none')
        self.fc2 = nn.Linear(config["intermediate_size"], config["embedding_size"])
        self.dropout = nn.Dropout(config["hidden_dropout_prob"])
        
    def forward(self, x):
        x = self.dropout(self.fc2(self.gelu(self.fc1(x))))
        return x
    
class Block(nn.Module):
    """
    Creating a basic transformer block using MLP and MHA.
    Also including the skip connections and the normalizations.
    """
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.layerNorm1 = nn.LayerNorm(config["embedding_size"])
        self.mlp = MLP(config)
        self.layerNorm2 = nn.LayerNorm(config["embedding_size"])
        
    def forward(self, x, output_attentions=False):
        # Self-attention
        attention_output, attention_probs = self.attention(self.layerNorm1(x), output_attentions=output_attentions)
        x = x + attention_output    # Skip connection

        # Feed Forward Net
        mlp_output = self.mlp(self.layerNorm2(x))
        x = x + mlp_output          # Skip connection
        
        # Return the transformer block's output and the attention probabilities (optional)
        if not output_attentions:
            return (x, None)
        else:
            return (x, attention_probs)
        
        
# Creating the encoder by stacking multiple blocks
class Encoder(nn.Module):
    """
    Sequentially stacking multiple transformer layers.
    """
    
    def __init__(self, config):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(config["num_hidden_layers"]):
            block = Block(config)
            self.blocks.append(block)
            
    def forward(self, x, output_attentions=False):
        # Calculate the transformer block's output for each block.
        all_attentions = []     # Empty list to store all the attention probabilities.
        for block in self.blocks:
            x, attention_probs = block(x, output_attentions=output_attentions)
            if output_attentions:
                all_attentions.append(attention_probs)
                
        # Return the encoder's outputs and attention probabilites.
        if not output_attentions:
            return (x, None)
        else:
            return (x, all_attentions)
        
# Creating the ViT. (finally)
class ViTForClassification(nn.Module):
    """
    The ViT model for classification
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = config["image_size"]
        self.embedding_size = config["embedding_size"]
        self.num_classes = config["num_classes"]
        # Creating the embedding module
        self.embedding = Embeddings(config)
        # Creating the encoder
        self.encoder = Encoder(config)
        # Create linear layer to project 'embedding_size' to 'num_classes'
        self.classifier = nn.Linear(self.embedding_size, self.num_classes)
        # Initialize the weights
        self.apply(self._init_weights)
        
    def forward(self, x, output_attentions=False):
        # Calculate the embedding output
        embedding_output = self.embedding(x)
        # Calculate the encoder's output
        encoder_output, all_attentions = self.encoder(embedding_output, output_attentions=output_attentions)
        # Calculate the logits
        logits = self.classifier(encoder_output[:, 0])  # For all batches, take the first embedding i.e. [CLS] token
        # Return the logits and attention probabilities
        if output_attentions:
            return (logits, None)
        else:
            return (logits, all_attentions)
        
    # Setting up weight initialization -
    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config["initializer_range"])
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, Embeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config["initializer_range"],
            ).to(module.cls_token.dtype)