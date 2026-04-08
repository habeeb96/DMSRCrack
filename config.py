import os
import torch

# ==========================================
#             USER SETTINGS
# ==========================================
# GPU SETUP
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

# PATHS
BASE_DIR = '/home/radhwan.saleh/surface_detection/datasets/crack1'
TRAIN_IMG_DIR = os.path.join(BASE_DIR, 'train', 'images')
TRAIN_MASK_DIR = os.path.join(BASE_DIR, 'train', 'masks')
VAL_IMG_DIR = os.path.join(BASE_DIR, 'test', 'images')
VAL_MASK_DIR = os.path.join(BASE_DIR, 'test', 'masks')

# HYPERPARAMETERS
TARGET_CLASSES = ['DeepCrack'] 
EPOCHS = 200
BATCH_SIZE = 32       
LEARNING_RATE = 3e-4  
WEIGHT_DECAY = 1e-5

# OUTPUTS
SAVE_NAME = "best_dmsrcrack_DeepCrack.pth"
CSV_LOG_NAME = "training_metrics_DeepCrack.csv"

# SYSTEM
SEED = 1234
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_WORKERS = 4
