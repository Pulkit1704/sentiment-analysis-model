import tensorflow as tf 


def load_data(filepath: str, train_test_ratio: float): 
    data = tf.data.Dataset.load(filepath, compression = "GZIP") 

    return train_test_validation_split(data, train_test_ratio) 

def train_test_validation_split(data: tf.data.Dataset, train_test_ratio: float,shuffle = True):

    if train_test_ratio > 1: 
        print("invalid ratio") 
    
    if shuffle: 
        data = data.shuffle(buffer_size=len(data), reshuffle_each_iteration=False) 

    
    train_set = data.take(int(len(data) * train_test_ratio)) 
    test_set = data.skip(int(len(data) * train_test_ratio)) 

    return train_set, test_set
