from dataset import *
from keras import backend as K
from model.hierarchical_attention_network import HNATT

# Seed
np.random.seed(1024)
from numpy.random import seed
seed(1024)
from tensorflow import set_random_seed
set_random_seed(1024)

dataset = "yelp"
batch_size = 4
epochs = 8
YELP_DATA_PATH = 'input/yelp/dataset/yelp_2018_1569264-docs.json'
SAVED_MODEL_DIR = 'saved_models'
SAVED_MODEL_FILENAME = 'model.h5'

(train_x, train_y), (test_x, test_y) = load_data_yelp(path=YELP_DATA_PATH, size=1569264)


K.clear_session()
h = HNATT()	

h.train(train_x, 
        train_y,        
		batch_size=batch_size,
		epochs=epochs,
		embeddings_path=True, 
		saved_model_dir=SAVED_MODEL_DIR,
		saved_model_filename=SAVED_MODEL_FILENAME)
