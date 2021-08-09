# -*- encoding:utf -*-
import torch


def save_model(model, model_path):
    if hasattr(model, "module"):
        torch.save(model.module.state_dict(), model_path)
    else:
        torch.save(model.state_dict(), model_path)


def load_model(model, model_path, strict=False):
    if hasattr(model, "module"):
        model.module.load_state_dict(torch.load(model_path, map_location='cpu'), strict=strict)
    else:
        model.load_state_dict(torch.load(model_path, map_location='cpu'), strict=strict)
    return model
