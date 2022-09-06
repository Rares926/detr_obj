import torch 
import argparse 


def clean(checkpoint):

    to_delete = []
    for key in checkpoint['model']:
        if 'transformer' not in key:
            to_delete.append(key)
    
    for key in to_delete:
        del checkpoint['model'][key]

    return checkpoint


