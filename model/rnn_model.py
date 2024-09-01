import tensorflow as tf 


class RNNModel(tf.keras.Model): 

    def __init__(self, vocabulary_size = 1000): 
        super().__init__()
        self.vocab_size = vocabulary_size

        self.textEncoder = tf.keras.layers.TextVectorization(max_tokens=self.vocab_size) 

        self.embedding = tf.keras.layers.Embedding(input_dim = self.vocab_size + 1 , output_dim = 64, mask_zero = True) 
        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units = 64))
        self.dense1 = tf.keras.layers.Dense(units = 32, activation = 'relu')
        self.dense2 = tf.keras.layers.Dense(units = 1) 
        
    def build(self, input_shape): 

        self.textEncoder.build(input_shape) 
        input_shape = self.textEncoder.compute_output_shape(input_shape) 

        self.embedding.build(input_shape) 
        input_shape = self.embedding.compute_output_shape(input_shape) 

        self.lstm.build(input_shape) 
        input_shape = self.lstm.compute_output_shape(input_shape) 

        self.dense1.build(input_shape) 
        input_shape = self.dense1.compute_output_shape(input_shape) 

        self.dense2.build(input_shape) 
        input_shape = self.dense2.compute_output_shape(input_shape) 

        self.built = True 

    def call(self, x): 
        x = self.textEncoder(x)

        x = self.lstm(self.embedding(x)) 

        output = self.dense2(self.dense1(x)) 
        return output 

    def train_encoder(self, train_data: tf.data.Dataset):

        self.textEncoder.adapt(train_data.map(lambda text, labels: text)) 
    
