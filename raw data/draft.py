import tensorflow as tf
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class LightGCN(tf.keras.Model):
    def __init__(self, data_config):
        super(LightGCN, self).__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.norm_adj = self.convert_sparse_matrix_to_sparse_tensor(data_config['norm_adj'])
        self.emb_dim = 64  # Example embedding dimension
        self.n_layers = 3  # Example number of layers
        self.lr = 0.001  # Learning rate
        self.decay = 0.0001  # Regularization parameter
        
        # Initialize embeddings
        self.user_embedding = tf.Variable(tf.random.normal([self.n_users, self.emb_dim]), trainable=True)
        self.item_embedding = tf.Variable(tf.random.normal([self.n_items, self.emb_dim]), trainable=True)

    def call(self, inputs):
        users, pos_items, neg_items = inputs
        return self.create_bpr_loss(users, pos_items, neg_items)

    def create_bpr_loss(self, users, pos_items, neg_items):
        u_embeddings = tf.nn.embedding_lookup(self.user_embedding, users)
        pos_i_embeddings = tf.nn.embedding_lookup(self.item_embedding, pos_items)
        neg_i_embeddings = tf.nn.embedding_lookup(self.item_embedding, neg_items)
        
        pos_scores = tf.reduce_sum(tf.multiply(u_embeddings, pos_i_embeddings), axis=1)
        neg_scores = tf.reduce_sum(tf.multiply(u_embeddings, neg_i_embeddings), axis=1)
        
        maxi = tf.math.log_sigmoid(pos_scores - neg_scores)
        mf_loss = -tf.reduce_mean(maxi)
        
        regularizer = tf.nn.l2_loss(u_embeddings) + tf.nn.l2_loss(pos_i_embeddings) + tf.nn.l2_loss(neg_i_embeddings)
        reg_loss = self.decay * regularizer
        
        return mf_loss + reg_loss

    def convert_sparse_matrix_to_sparse_tensor(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data.astype(np.float32), coo.shape)

def train_lightgcn(data_config, user_to_index, item_to_index, epochs=100, batch_size=256):
    model = LightGCN(data_config)
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.lr)

    for epoch in range(epochs):
        total_loss = 0
        num_batches = (len(user_to_index) + batch_size - 1) // batch_size

        for batch_idx in range(num_batches):
            users, pos_items, neg_items = sample_training_data(list(user_to_index.keys()), list(item_to_index.keys()), user_to_index, item_to_index, batch_size)
            users, pos_items, neg_items = np.array(users), np.array(pos_items), np.array(neg_items)

            with tf.GradientTape() as tape:
                loss = model([users, pos_items, neg_items])
            
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            total_loss += loss / num_batches
        
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}')

    return model, model.user_embedding.numpy(), model.item_embedding.numpy()

def sample_training_data(user_ids, item_ids, user_to_index, item_to_index, batch_size):
    users, pos_items, neg_items = [], [], []
    for _ in range(batch_size):
        user = np.random.choice(user_ids)
        pos_item = np.random.choice(item_ids)
        neg_item = np.random.choice([item for item in item_ids if item != pos_item])
        users.append(user_to_index[user])
        pos_items.append(item_to_index[pos_item])
        neg_items.append(item_to_index[neg_item])
    return users, pos_items, neg_items

# Load data and preprocess
data_config, user_to_index, item_to_index = load_and_preprocess_data("data.csv")

# Train the model
model, user_embeddings, item_embeddings = train_lightgcn(data_config, user_to_index, item_to_index, epochs=100, batch_size=128)

def convert_embeddings_to_dict(embeddings_array, index_to_id_map):
    """ Convert embeddings array to a dictionary with ids as keys. """
    return {index_to_id_map[idx]: embeddings_array[idx] for idx in range(len(embeddings_array))}

# After training, convert embeddings to dictionaries
user_embeddings_dict = convert_embeddings_to_dict(user_embeddings, {v: k for k, v in user_to_index.items()})
item_embeddings_dict = convert_embeddings_to_dict(item_embeddings, {v: k for k, v in item_to_index.items()})


# Example of recommendation function
def recommend_items(user_embeddings, item_embeddings, user_id, user_to_index, item_to_index, top_k=10):
    if user_id not in user_to_index:
        print(f"User ID {user_id} not found.")
        return []
    user_index = user_to_index[user_id]
    user_embedding = user_embeddings[user_index]
    
    # Create a reverse dictionary to map index back to item ID
    index_to_item = {v: k for k, v in item_to_index.items()}

    scores = {index_to_item[item_idx]: np.dot(embed, user_embedding) for item_idx, embed in enumerate(item_embeddings)}
    top_k_items = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return top_k_items

def recommend_users(item_embeddings, user_embeddings, item_id, item_to_index, user_to_index, top_k=10):
    if item_id not in item_to_index:
        print(f"Item ID {item_id} not found.")
        return []
    item_index = item_to_index[item_id]
    item_embedding = item_embeddings[item_index]

    # Create a reverse dictionary to map index back to user ID
    index_to_user = {v: k for k, v in user_to_index.items()}

    scores = {index_to_user[user_idx]: np.dot(embed, item_embedding) for user_idx, embed in enumerate(user_embeddings)}
    top_k_users = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return top_k_users


# Get recommendations
top_k_items = recommend_items(user_embeddings, item_embeddings, 1000, user_to_index, item_to_index, top_k=5)
print('Top 5 recommended items for user 1000:', top_k_items)

def convert_embeddings_to_dict(embeddings_array, index_to_id_map):
    """ Convert embeddings array to a dictionary with ids as keys. """
    return {index_to_id_map[idx]: embeddings_array[idx] for idx in range(len(embeddings_array))}

import numpy as np

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))



# After training, convert embeddings to dictionaries
user_embeddings_dict = convert_embeddings_to_dict(user_embeddings, {v: k for k, v in user_to_index.items()})
item_embeddings_dict = convert_embeddings_to_dict(item_embeddings, {v: k for k, v in item_to_index.items()})


# Assuming `user_embeddings` is a dictionary with user_id as keys and embeddings as values
print(cosine_similarity(user_embeddings_dict[1000], user_embeddings_dict[2000]))


# TensorFlow version: 2.15.0
# NumPy version: 1.25.2
# SciPy version: 1.11.4
# Pandas version: 2.0.3
