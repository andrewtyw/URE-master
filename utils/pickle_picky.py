import  pickle

def save(obj,path_name):
    """
        save a object to a pickle file
    """
    print("save file to:",path_name)
    with open(path_name,'wb') as file:
        pickle.dump(obj,file)

def load(path_name: object) -> object:
    """
        to load a stored object
    """
    with open(path_name,'rb') as file:
        return pickle.load(file)