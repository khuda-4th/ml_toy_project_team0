import torch
import numpy as np
import pandas as pd
import fasttext
import fasttext.util
import joblib
from price_prediction_model import MLP
from ingredient_prediction_model import MultiClassModel

def word_embedding(word_name):
    model_path = "model/word_embedding_model.bin"
    word_model = fasttext.load_model(model_path)
    return word_model.get_word_vector(word_name)

def encode_region(region):
    possible_regions = ["BS", "CC", "DG", "DJ", "GJ", "GK", "IN", "JJ", "JR", "KS", "KW", "SL", "US"]

    if region not in possible_regions:
        raise ValueError("Invalid region. Please choose from: BS, CC, DG, DJ, GJ, GK, IN, JJ, JR, KS, KW, SL, US")

    region_index = possible_regions.index(region)

    one_hot = [0] * len(possible_regions)
    one_hot[region_index] = 1

    return one_hot

def concatenate_feature(word_vec, food_type_vec, food_subtype_vec, region_vec, ingred_vec):
    model_input = np.hstack((word_vec, food_type_vec, food_subtype_vec, region_vec, ingred_vec))
    assert model_input.shape[0] == 400
    return model_input


def get_food_type(word_vec):
    class_num = 7
    with open('model/food_type_prediction_model.pkl', 'rb') as model_file:
        loaded_model = joblib.load(model_file)
    
    predicted_labels = loaded_model.predict([word_vec])
    
    idx = predicted_labels  
    modified_labels = np.zeros(class_num)
    modified_labels[idx] = 1

    return modified_labels


def get_food_subtype(word_vec):
    class_num = 30
    with open('model/food_subtype_prediction_model.pkl', 'rb') as model_file:
        loaded_model = joblib.load(model_file)
    
    predicted_labels = loaded_model.predict([word_vec])  
    
    idx = predicted_labels  
    modified_labels = np.zeros(class_num)
    modified_labels[idx] = 1

    return modified_labels

def get_ingred(word_vec):
    ingred_tensor_input = torch.FloatTensor(word_vec)
    ingred_model = MultiClassModel(300, 50)
    ingred_checkpoint = torch.load('model/ingredient_prediction_model.pth')
    ingred_model.load_state_dict(ingred_checkpoint['model_state_dict'])

    ingred_model.eval()
    with torch.no_grad():
        output = ingred_model(ingred_tensor_input)
    
    return output


def get_price(model_input):
    tensor_input = torch.FloatTensor(model_input)
    checkpoint = torch.load('model/price_prediction_model.pth')
    price_model = MLP(400)
    price_model.load_state_dict(checkpoint['model_state_dict'])

    price_model.eval()
    with torch.no_grad():
        output = price_model(tensor_input)
    
    return output.item()


def inference():
    food_name = input("type food_name: ")
    region = input("type among BS, CC, DG, DJ, GJ, GK, IN, JJ, JR, KS, KW, SL, US: ")
    word_vec = word_embedding(food_name)
    region_vec = encode_region(region)
    food_type_vec = get_food_type(word_vec)
    food_subtype_vec = get_food_subtype(word_vec)
    ingred_vec = get_ingred(word_vec)
    
    model_input = concatenate_feature(word_vec, food_type_vec, food_subtype_vec, region_vec, ingred_vec)
    price = get_price(model_input)
    print(f"proper price is {int(price / 100) * 100} won...")
    return 0

inference()