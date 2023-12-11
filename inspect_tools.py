from contextlib import contextmanager
from bmtrain import CheckpointBlock
import sys
import torch
log_file = set()
@contextmanager
def custom_redirection(fileobj):
    if isinstance(fileobj, str):
        if fileobj not in log_file:
            ftmp = open(fileobj,"w")
            ftmp.close()
            log_file.add(fileobj)
        file_handle = open(fileobj,"a") 
    else:
        file_handle = fileobj
    old = sys.stdout
    sys.stdout = file_handle 
    try:
        yield file_handle 
    finally:
        sys.stdout = old
        file_handle.close()

def look_var(layer, _, output):
    try:
        print(f"{layer.__name__}: {output.min()}")
    except:
        print(f"{layer.__name__}: {output[0].min()}")


def look_inp_weight(look_inp,look_weight):
    def look_inp_func(layer, inp):
        if look_inp:
            try:
                print(f"{layer.__name__}: {inp.min()}")
            except:
                print(f"{layer.__name__}: {inp[0].min()}")
        if look_weight:
            print(f"{layer.__name__} weight: {layer._parameters}")
    return look_inp_func

def check_grad(layer, grad_input, grad_output):
    # 检查输出梯度
    for i, grad in enumerate(grad_output):
        if grad is not None:
            if torch.isnan(grad).any():
                print(f'NaN detected in grad_output[{i}] of layer {layer.__name__}')

    # 检查输入梯度
    for i, grad in enumerate(grad_input):
        if grad is not None:
            if torch.isnan(grad).any():
                print(f'NaN detected in grad_input[{i}] of layer {layer.__name__}')

def lookup_output(model,layers=set(), look_input=False, look_weight=False):
    for key,layer in model.named_modules():
        layer.__name__ = key
        if layer not in layers:
            layers.add(layer)
        else:
            continue
        if len(layer._modules) !=0:
            layer.register_forward_hook(look_var)
            # layer.register_backward_hook(check_grad)
            lookup_output(layer,layers,look_input=look_input,look_weight=look_weight)
            layer.register_forward_pre_hook(look_inp_weight(look_input,look_weight))
        else:
            layer.register_forward_hook(look_var)
            # layer.register_backward_hook(check_grad)
            layer.register_forward_pre_hook(look_inp_weight(look_input,look_weight))


