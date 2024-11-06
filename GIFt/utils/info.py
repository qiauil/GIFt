import torch.nn as nn
from typing import Sequence
import os
import copy

def collect_trainable_parameters(module:nn.Module):
    index=0
    num_para_after=0
    fine_tuning_parameters=[]
    for name,p in module.named_parameters():
        if p.requires_grad:
            p_num=p.numel()
            num_para_after+=p_num
            fine_tuning_parameters.append([str(index),name,str(list(p.shape)),str(p_num)])
            index+=1
    return fine_tuning_parameters,num_para_after

def table_info(table:Sequence[Sequence],header:Sequence[str],return_hline=False)->str:
    table=copy.deepcopy(table)
    table.insert(0,header)
    col_widths = [max(len(str(item)) for item in col) for col in zip(*table)]
    h_line="-".join("-"*width for width in col_widths)
    str_table=[]
    for row in table:
        str_table.append(" | ".join(str(item).ljust(width) for item, width in zip(row, col_widths)))
    str_table.insert(1, h_line)
    str_table.insert(0, h_line)
    str_table.append(h_line)
    if return_hline:
        return os.linesep.join(str_table),h_line
    else:
        return os.linesep.join(str_table)