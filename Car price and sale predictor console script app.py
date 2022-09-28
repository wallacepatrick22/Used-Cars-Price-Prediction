#import needed Libraries
import pandas as pd
import numpy as np
import os.path as Path
import time
from pathlib import Path
from os import system, name
from termcolor import colored
import pickle
from sklearn.preprocessing import StandardScaler
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Get original dataframe to enable parameter transformation         
df = pd.read_csv("C:/Users/USCHIP/Car Sales Project/UserCarData_no_bracket.csv")

# Define initial parameters
needed_params = ["fuel_type", "city", "km_driven", "engine_capacity", "region", "max_power", "year", "torque_in_nm", "name"]
txt_confirm = []
num_confirm = []
user_info_list = []
user_info_dict = {}

# Define clear and process delay function
def clear():
    # for windows
    if name == 'nt':
        _ = system("cls")
  
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system("clear")

def get_process(processed):
    dot = "."
    process = processed
    for i in range(3):
        time.sleep(1)
        clear()
        processed = processed + dot
        print(processed)

def get_completed(process_val):
    time.sleep(1)
    print(process_val.capitalize(),"completed!")
    time.sleep(1)
    clear()
    
def case_transform(column):
    for trans_value in df[column].unique():
        df[column].replace(trans_value, trans_value.lower(), inplace=True)
    return df[column]
    
def get_end_prog():
    _ = system("exit")

# Loading my preferred model which is the stacked models using the stacked classifier and regressor as applicable
def get_models():
    price_path = "C:/Users/USCHIP/Car Sales Project/Saved Model/Car_Price_Estimator_Model.pkl"
    sold_path =  "C:/Users/USCHIP/Car Sales Project/Saved Model/Car_Sold_Classifier_Model.pkl"

    model = []
    path_list = [price_path, sold_path]
    model_name = ["Model[0]", "Model[1]"]

    for i,j in zip(path_list,model_name):
        filepath = Path(i)
        stacked_model = pickle.load( open(filepath, "rb"))
        model.append(stacked_model)
        print("\nNOTE: Loaded {} model as : {} successfully".format(i, j))
    return model

# Sort out scaling procedure
def init_scale_sequence():
    # Get data to fit our standard scaler
    refined_car_df = pd.read_csv("C:/Users/USCHIP/Car Sales Project/New_Guide_Sheets/CSV_file/features.csv")
    refined_car_df.drop("index", axis=1, inplace=True)
    return refined_car_df

# Function to collect input parameters
# Function to collect, prepare and evaluate user data before making predictions using a model.
def collect_input_params():
    for i in range(len(needed_params)):
        
        user_input = input("Please enter {} : ".format(needed_params[i]))
        
        # Data validation
        if i in [0, 1, 4]:
            try:
                user_input == float(user_input)
                print("You have just entered a wrong data type, please enter a text not a number\n You can close the app and start again")
                get_end_prog()
            except:
                user_input = user_input.lower()
                user_info_dict[needed_params[i]] = user_input
                
                
        elif i in [2, 3, 5, 6, 7]:
            try:
                user_input == float(user_input)
                if i in [2,3,6]:
                    user_input = float(user_input)
                elif i in [5,7]:
                    user_input = int(user_input)
                user_info_dict[needed_params[i]] = user_input

            except:
                print("You have just entered a wrong data type, please enter a number not a text\n You can close the app and start again")
                get_end_prog()
       
        elif i in [8]:
            try:
                user_input == float(user_input)
                print("You have just entered a wrong data type, please enter a text not a number\n You can close the app and start again")
                get_end_prog()
            except:
                user_input = user_input.title()
                user_info_dict[needed_params[i]] = user_input
                
    
    get_pred = "Coverting Texts"
    get_process(get_pred)
    
    # convert user input into a comprehensive dataframe
    df_gen = pd.DataFrame.from_dict(user_info_dict, orient="index").T
        
    # Code to convert text parameters to numbers as was done in the training data
    fuel_data = ["diesel","petrol","lpg","cng"]
    try:
        conv_count = 0
        if user_info_dict["fuel_type"].lower() in fuel_data:
            for e,i in enumerate(fuel_data):
                i = i.lower()
                df_gen.fuel_type.replace(i, e, inplace=True)
        
        conv_count = 1
        df["City"] = case_transform("City")
        if user_info_dict["city"].lower() in df["City"].unique():
            for e,i in enumerate(df["City"].unique()):
                i = i.lower()
                df_gen["city"].replace( i, e, inplace=True)
        
        conv_count = 2
        df["Region"] = case_transform("Region")
        if user_info_dict["region"].lower() in df["Region"].unique():
            for e,i in enumerate(df["Region"].unique()):
                i = i.lower()
                df_gen["region"].replace( i, e, inplace=True)
    except:
        if conv_count == 0:
            print("You have entered a text not existing in our records for fuel type, kindly cofirm the fuel type you entered and try again or contact the admin.")
            time.sleep(1)  
        elif conv_count == 1:
            print("You have entered a text not existing in our records for city, kindly cofirm the city you entered and try again or contact the admin.")
            time.sleep(1)
        elif conv_count == 2:
            print("You have entered a text not existing in our records for region, kindly cofirm the region you entered and try again or contact the admin.")
            time.sleep(1)
            
    # Create dataframe for the selling price prediction parameters and convert the values to an array    
    df1  = df_gen[["torque_in_nm", "max_power" , "engine_capacity", "year"]]
    df_p = pd.concat([df1,df1], axis=0)
    df_p = df_p.values
    
    # Initialise data scaling procedure using standard scaler    
    features_df = init_scale_sequence()
    # Fitting and transforming the imported csv data using standard scaler scaler so as to transform our new data
    standard_scaler = StandardScaler()
    standard_scaler.fit_transform(features_df)   
    
    # Create dataframe for the sold or not sold prediction parameters and convert the values to an array  
    df2  = df_gen[["fuel_type", "city", "km_driven", "engine_capacity", "region"]]
    df_s = pd.concat([df2,df2], axis=0)
    df_s = df_s.values
    df_s = standard_scaler.transform(df_s)
    
    clear()
    get_pred = "Getting Prediction"
    get_process(get_pred)
    
    # Making prediction with stacking regressor model for price
    predicted_price = xgboost_reg_model.predict(df_p[0:1])[0]
    
    # Making prediction with stacking classifier model for easy  sold or not easily sold
    predicted_sale = stacked_opt_clf_model.predict(df_s[0:1])[0]
    
    get_completed(get_pred)
    
    # displaying the result of the predictions
    if buy_sell.lower() == "buy":
        print("The {} car you want to {} has a predicted price of {:.2f} dollars and a tolerance range of between {:.2f} dollars and {:.2f} dollars".format(df_gen.name.unique(), buy_sell, predicted_price, predicted_price*0.9, predicted_price*1.1))
        time.sleep(5)
    elif buy_sell.lower() == "sell":
        if predicted_sale == 0:
            print("The {} car you want to {} has a predicted price of {:.2f} dollars and a tolerance range of between {:.2f} dollars and {:.2f} dollars and will not be easily sold".format(df_gen.name.unique(), buy_sell, predicted_price, predicted_price*0.9, predicted_price*1.1))
            time.sleep(5)
        elif predicted_sale == 1:
            print("The {} car you want to {} has a predicted price of {:.2f} dollars and a tolerance range of between {:.2f} dollars and {:.2f} dollars and will be easily sold".format(df_gen.name.unique(), buy_sell, predicted_price, predicted_price*0.9, predicted_price*1.1))
            time.sleep(5)
            
# Initialisation of program
print("HELLO! AND WELCOME TO A WALLACE TUDEME PROJECT")
time.sleep(1)
print("This project helps to predict car price and if a car would be sold or not sold\n\n")

load_app = "Loading Application"
get_process(load_app)
get_completed(load_app)

# converting loaded models into suited names
getting_models = "Getting Models"
get_process(getting_models)

model  = get_models()
xgboost_reg_model = model[0]
stacked_opt_clf_model = model[1]

get_completed(getting_models)

# Request user input for buy or sell
print(colored("NOTE: Please enter {} or {} below and hit the enter button: ".format("[buy]","[sell]"), color = "red", on_color="on_white", attrs=['bold', 'blink']))
buy_sell = str(input("Do you want to buy or sell a car? : "))
try:
    buy_sell == float(buy_sell)
    print("You have just entered a wrong data type, please enter a text not a number\n You can close the app and start again")
except:
    if buy_sell.lower() in ["buy", "sell"]:
        buy_sell = buy_sell.title()
    else:
        print("You filled in a wrong text, please fill in the appropraite text.")

# Get and Process user data
collect_input_params()