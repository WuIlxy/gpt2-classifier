import torch
import torch.nn as nn
import math
from transformers import GPT2LMHeadModel, GPT2Config, GPT2TokenizerFast

class CustomGPT2(nn.Module):
    """
    Task 1: Custom GPT2 model with random initialization
    """
    def __init__(self, tokenizer):
        super().__init__()
        # Get GPT2 config but ensure random initialization
        self.config = GPT2Config.from_pretrained('gpt2')
        self.config._from_pretrained = False

        # Initialize model with random weights
        self.model = GPT2LMHeadModel(self.config)

        # Resize token embeddings to match our tokenizer
        self.model.resize_token_embeddings(len(tokenizer))

        # Ensure output is log probabilities
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return self.log_softmax(outputs.logits)

class LoRALayer(nn.Module):
    """
    Low Rank Adapter layer implementation for fine-tuning
    """
    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        # Initialize LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Initialize weights using kaiming initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling

class CustomGPT2Task5(nn.Module):
    """
    Extended version of CustomGPT2 for Task 5
    """
    def __init__(self, tokenizer):
        super().__init__()
        # Get GPT2 config but ensure random initialization
        self.config = GPT2Config.from_pretrained('gpt2')
        self.config._from_pretrained = False

        # Initialize model with random weights
        self.model = GPT2LMHeadModel(self.config)

        # Resize token embeddings to match our tokenizer
        self.model.resize_token_embeddings(len(tokenizer))

        # Ensure output is log probabilities
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def resize_token_embeddings(self, new_num_tokens):
        # Method for Task 5
        return self.model.resize_token_embeddings(new_num_tokens)

    def forward(self, input_ids, attention_mask=None):
        outputs = self.model(input_ids, attention_mask=attention_mask)
        return self.log_softmax(outputs.logits)

class AgeClassificationModelWithLoRA(nn.Module):
    """
    Task 5: Age classification model using LoRA fine-tuning
    """
    def __init__(self, model_path, token, lora_rank=8, lora_alpha=16):
        super().__init__()

        # Initialize tokenizer with [cls] token
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        special_tokens = {'additional_special_tokens': ['[cls]']}
        self.tokenizer.add_special_tokens(special_tokens)

        # Initialize base model
        self.base_model = CustomGPT2Task5(self.tokenizer)

        # Resize token embeddings to include new special token
        self.base_model.resize_token_embeddings(len(self.tokenizer))

        # Load pretrained weights
        try:
            from huggingface_hub import hf_hub_download
            checkpoint_path = hf_hub_download(
                repo_id=model_path,
                filename="pytorch_model.bin",
                token=token
            )
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            filtered_state_dict = {k: v for k, v in state_dict.items()
                                 if k in self.base_model.state_dict() and
                                 v.shape == self.base_model.state_dict()[k].shape}
            self.base_model.load_state_dict(filtered_state_dict, strict=False)
            print("Successfully loaded pretrained weights")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Continuing with base model weights...")

        # Add LoRA layers to attention modules
        self.lora_layers = nn.ModuleDict()
        for name, module in self.base_model.model.named_modules():
            if isinstance(module, nn.Linear) and "attn" in name:
                self.lora_layers[name] = LoRALayer(
                    module.in_features,
                    module.out_features,
                    rank=lora_rank,
                    alpha=lora_alpha
                )

                # Create new forward method
                def make_forward(original_module, lora_layer):
                    def forward(x):
                        return original_module(x) + lora_layer(x)
                    return forward

                # Bind the new forward method
                module.forward = make_forward(module, self.lora_layers[name]).__get__(module, type(module))

        # Two-layer MLP classification head as specified
        hidden_size = self.base_model.model.config.n_embd
        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 20)  # 20 classes for ages 1-20
        )

        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze LoRA parameters and classification head
        for lora in self.lora_layers.values():
            for param in lora.parameters():
                param.requires_grad = True
        for param in self.classification_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        last_hidden_state = outputs.hidden_states[-1]
        cls_token_state = last_hidden_state[:, -1, :]  # Get the last token ([cls])

        age_logits = self.classification_head(cls_token_state)
        return age_logits
