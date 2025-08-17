import torch.nn as nn


def get_network_data(model, layer_name, attribute):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(x in name for x in ['to_q', 'to_k', 'to_v']):
            print(">>>", name, module.__class__.__name__)
            model = model
            module_keys = name.split('.')
            for key in module_keys:
                if key.isdigit():
                    model = model[int(key)]
                else:
                    model  = getattr(model, key)

            print(model)

def register_hook(model):
    for name, module in model.named_modules:
        print(">>>", name, module.__class__.__name__)


# def hook_function(model, input, output):
