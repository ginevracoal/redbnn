import os
import pickle as pkl


def save_to_pickle(data, path, filename):

    full_path=os.path.join(path, filename+".pkl")
    print("\nSaving pickle: ", full_path)
    os.makedirs(path, exist_ok=True)
    with open(full_path, 'wb') as f:
        pkl.dump(data, f)

def load_from_pickle(path, filename):

    full_path=os.path.join(path, filename+".pkl")
    print("\nLoading from pickle: ", full_path)
    with open(full_path, 'rb') as f:
        u = pkl._Unpickler(f)
        u.encoding = 'latin1'
        data = u.load()

    return data
    
def unpickle(file):
    """ Load byte data from file"""
    with open(file, 'rb') as f:
        data = pkl.load(f, encoding='latin-1')
    return data
