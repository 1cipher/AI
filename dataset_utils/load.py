from sklearn.datasets import load_files

def loadSet():

    train_set = load_files(r'C:\Users\Utente\Desktop\AI\train')
    train_data,train_target = train_set.data, train_set.target
    print('train dataset caricati')

    test_set = load_files(r'C:\Users\Utente\Desktop\AI\test')
    test_data,test_target = test_set.data,test_set.target
    print('test dataset caricati')

    return train_data,train_target,test_data,test_target