# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 14:57:13 2026

@author: Carles
"""

import sys

def print_bold_old(text):
    # Check if the code is running inside a Jupyter Notebook
    if 'ipykernel' in sys.modules:
        from IPython.display import display, HTML
        display(HTML(f"<b>{text}</b>"))
    else:
        # Fallback for standard terminal/command prompt
        print(f"\033[1m{text}\033[0m")
        
        
        
def print_bold(text):
    # Check if we are running in an environment that can actively render HTML
    try:
        from IPython import get_ipython
        ip = get_ipython()
        
        # Ensure an IPython session exists and supports rich frontend rendering (like Jupyter)
        if ip is not None and 'ZMQInteractiveShell' in ip.__class__.__name__:
            from IPython.display import display, HTML
            display(HTML(f"<b>{text}</b>"))
            return
    except ImportError:
        pass

    # Fallback for Spyder console, standard terminal, or command prompt
    print(f"\033[1m{text}\033[0m")
    
    
def get_config_structure(hidden_struct, config=None):
    struct = ''
    pos = 0 
    
    for n_layer, n_per_layer in enumerate(hidden_struct):
        
        struct += config[pos:pos + n_per_layer]
        
        if n_layer != len(hidden_struct) - 1:
            struct += '-'
        
        pos += n_per_layer 
            
    return struct

