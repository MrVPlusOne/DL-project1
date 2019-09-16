from pathlib import Path
from PIL import Image
import numpy as np

def get_data_loader(path_name, batch_size=1):
    path = Path(path_name)
    
    def _loader():
        for img_path in path.glob('*.jpg'):
            img = Image.open(img_path)
            yield img
            
    return _loader

def grade(module, bottleneck_limit=8192):
    encode, decode = module.encode, module.decode
    
    data_loader = get_data_loader('data')
    loss = []
    
    for img in data_loader():
        x = encode(img)
        assert x.nbytes <= bottleneck_limit, "Bottleneck too large!"
        
        reconstructed_img = np.array(decode(x)).astype(float)
        assert reconstructed_img.shape == (256,256,3), "Output resolution wrong!"
        
        loss.append(np.abs(np.array(img).astype(float) - reconstructed_img))
    
    print ("Loss: %.3f"%np.mean(loss))
    

def run():
    import argparse
    import importlib
    
    parser = argparse.ArgumentParser()
    parser.add_argument('assignment', help='path of assignment')
    
    args = parser.parse_args()
    
    module = importlib.import_module(args.assignment)
    
    grade(module)
