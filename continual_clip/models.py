from omegaconf import DictConfig

import clip
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig
import pdb

from .utils import get_class_ids_per_task, get_class_names


class ClassIncrementalCLIP(nn.Module):
    def __init__(self, cfg, device, jit=False):
        super().__init__()
        self.prompt_template = cfg.prompt_template
        self.device = device
        self.classes_names = None
        self.model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)
        # # pdb.set_trace()

        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            target_modules=["attn"],
            lora_dropout=cfg.lora_dropout,
        )
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        # pdb.set_trace()
        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names = []
        self.text_tokens = None
    
    def forward(self, image, test=False):
        if test:
            with torch.no_grad():
                logits_per_image, _ = self.model(image, self.text_tokens)
                probs = logits_per_image.softmax(dim=-1)
            return {"probs": probs}
        else:
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(self.text_tokens)

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            # shape = [global_batch_size, global_batch_size]
            return {
                    "logits_per_image": logits_per_image,
                    "logits_per_text": logits_per_text,
                    "image_features": image_features,
                    "text_features": text_features,
                    }

    def adaptation(self, task_id):
        self.current_class_names += get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)


class DomainIncrementalCLIP(nn.Module):
    def __init__(self, cfg, device, jit=False) -> None:
        super().__init__()
        self.model, self.transforms = clip.load(cfg.model_name, device=device, jit=jit)
        self.text_tokens = None
        self.prompt_template = cfg.prompt_template
        self.device = device

    def forward(self, image):
        with torch.no_grad():
            logits_per_image, _ = self.model(image, self.text_tokens)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        return probs

    def tokenize(self, class_names):
        self.text_tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in class_names]
        ).to(self.device)



class TaskAgnosticCLIP(nn.Module):
    pass



def load_model(cfg: DictConfig, device: torch.device) -> nn.Module:
    r"""Load a CLIP model in different continual scenarios.
    
    Arguments:
        cfg (DictConfig): Experiment configurations.
        device (torch.device): Device to train (or) evaluate the model on.
        
    Returns:
        nn.Module: Return scenario specific CLIP model.
    """
    if cfg.scenario == "class":
        return ClassIncrementalCLIP(cfg, device)
    elif cfg.scenario == "domain":
        return DomainIncrementalCLIP(cfg, device)
    elif cfg.scenario == "task-aganostic":
        return TaskAgnosticCLIP(cfg, device)
    else:
        raise ValueError(f"""
            `{cfg.scenarios}` is not a valid scenario, 
            Please choose from ['class', "domain', 'task-agnostic']
        """)

