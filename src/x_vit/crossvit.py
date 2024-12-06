"""CrossViT architecture to a regression task with dual images as inputs."""
import timm
import torch
import torch.nn as nn
from timm.models.crossvit import CrossAttentionBlock


class CrossViTDualBranch(nn.Module):
    def __init__(self, branch_model_name: str = "vit_base_patch16_224"):
        super().__init__()

        self.branch_model_name = branch_model_name
        
        # Load pretrained ViT models
        self.branch_1 = timm.create_model(branch_model_name, pretrained=True)
        self.branch_2 = timm.create_model(branch_model_name, pretrained=True)

        # Get embedding dimension
        self.embed_dim = self.branch_1.embed_dim

        # Remove the heads
        self.branch_1.head = nn.Identity()
        self.branch_2.head = nn.Identity()
        
        # Cross attention layers for each branch
        self.cross_attn_1 = nn.ModuleList([
            CrossAttentionBlock(dim=self.embed_dim, num_heads=8)
            for _ in range(len(self.branch_1.blocks))
        ])
        
        self.cross_attn_2 = nn.ModuleList([
            CrossAttentionBlock(dim=self.embed_dim, num_heads=8)
            for _ in range(len(self.branch_2.blocks))
        ])
        
        # Regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.embed_dim, 1)
        )

    def forward(self, img1, img2): #img (b, w, h, c)
        # Initial patch embeddings and setup for both branches
        x1 = self.branch_1.patch_embed(img1) # (b, 16**2 * 3, 224/16)
        x2 = self.branch_2.patch_embed(img2) # (b, 16**2 * 3, 224/16)
        
        # Add CLS tokens
        cls_1 = self.branch_1.cls_token.expand(x1.shape[0], -1, -1) # (B, 16**2 * 3)
        cls_2 = self.branch_2.cls_token.expand(x2.shape[0], -1, -1) # (B, 16**2 * 3)
        x1 = torch.cat((cls_1, x1), dim=1) # (b, 16**2 * 3, 224/16 + 1)
        x2 = torch.cat((cls_2, x2), dim=1) # (b, 16**2 * 3, 224/16 + 1)
        
        # Add position embeddings
        x1 = self.branch_1.pos_drop(x1 + self.branch_1.pos_embed) # (b, 16**2 * 3, 224/16 + 1)
        x2 = self.branch_2.pos_drop(x2 + self.branch_2.pos_embed) # (b, 16**2 * 3, 224/16 + 1)
        
        # Process through transformer blocks with cross attention at each layer 
        for i in range(len(self.branch_1.blocks)):
            # Regular self-attention for both branches
            x1 = self.branch_1.blocks[i](x1) # (b, N, hidden_dims)
            x2 = self.branch_2.blocks[i](x2) # (b, N, hidden_dims)
            
            # Extract current CLS tokens and patch tokens
            cls_1 = x1[:, 0:1] # (b, N, hidden_dims)
            cls_2 = x2[:, 0:1]
            patches_1 = x1[:, 1:]
            patches_2 = x2[:, 1:]
            
            # Cross attention: CLS token of each branch attends to patch tokens of other branch
            cross_attended_cls_1_x2 = self.cross_attn_1[i](torch.cat((cls_1, patches_2), dim=1))
            cross_attended_cls_2_x1 = self.cross_attn_2[i](torch.cat((cls_2, patches_1), dim=1))
            
            # Update CLS tokens
            x1 = torch.cat([cross_attended_cls_1_x2, patches_1], dim=1)
            x2 = torch.cat([cross_attended_cls_2_x1, patches_2], dim=1)
        
        # Final layer norm
        x1 = self.branch_1.norm(x1)
        x2 = self.branch_2.norm(x2)
        
        # Extract final CLS tokens
        cls_1 = x1[:, 0]
        cls_2 = x2[:, 0]
        
        # Combine features for regression
        combined_features = torch.cat([cls_1, cls_2], dim=-1)
        
        # Regression output
        return self.regression_head(combined_features)