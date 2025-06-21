import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Install TensorFlow if not already installed (only needed once)
# !pip install tensorflow

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
import gradio as gr
from PIL import Image