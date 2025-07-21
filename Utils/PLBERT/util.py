import os
import yaml
import torch
from transformers import AlbertConfig, AlbertModel

class CustomAlbert(AlbertModel):
    def forward(self, *args, **kwargs):
        # Handle inputs that are too long (exceeding 512 tokens)
        if len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].size(1) > 512:
            args = list(args)
            args[0] = args[0][:, :512]
            args = tuple(args)
        
        # Handle attention mask if provided as positional argument
        if len(args) > 1 and isinstance(args[1], torch.Tensor) and args[1].size(1) > 512:
            args = list(args)
            args[1] = args[1][:, :512]
            args = tuple(args)
            
        # Handle inputs passed as keyword arguments
        if 'input_ids' in kwargs and kwargs['input_ids'].size(1) > 512:
            kwargs['input_ids'] = kwargs['input_ids'][:, :512]
        if 'attention_mask' in kwargs and kwargs['attention_mask'].size(1) > 512:
            kwargs['attention_mask'] = kwargs['attention_mask'][:, :512]
        # Call the original forward method
        outputs = super().forward(*args, **kwargs)

        # Only return the last_hidden_state
        return outputs.last_hidden_state


def load_plbert(log_dir):
    config_path = os.path.join(log_dir, "config.yml")
    plbert_config = yaml.safe_load(open(config_path))
    
    albert_base_configuration = AlbertConfig(**plbert_config['model_params'])
    bert = CustomAlbert(albert_base_configuration)

    files = os.listdir(log_dir)
    ckpts = []
    for f in os.listdir(log_dir):
        if f.startswith("step_"): ckpts.append(f)

    iters = [int(f.split('_')[-1].split('.')[0]) for f in ckpts if os.path.isfile(os.path.join(log_dir, f))]
    iters = sorted(iters)[-1]

    checkpoint = torch.load(log_dir + "/step_" + str(iters) + ".t7", map_location='cpu')
    state_dict = checkpoint['net']
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        if name.startswith('encoder.'):
            name = name[8:] # remove `encoder.`
            new_state_dict[name] = v
    # Replace the problematic line:
    # del new_state_dict["embeddings.position_ids"]
    
    # With this safe check:
    if "embeddings.position_ids" in new_state_dict:
        del new_state_dict["embeddings.position_ids"]
    bert.load_state_dict(new_state_dict, strict=False)
    
    return bert