import pickle
import pandas as pd
import numpy as np

# load saved models
with open('outputs/model_clf.pkl', 'rb') as file:
    lr_clf = pickle.load(file)

with open('outputs/preprocess.pkl', 'rb') as file:
    preprocessor = pickle.load(file)

# load from allocate_data to test
from allocate_data import allocate_student_data
dataset_path, allocation = allocate_student_data(student_id="3312605")
MODEL_SEED = allocation["model_seed"]   

# Make the read test file does NOT have rating, by dropping the rating column
df = pd.read_csv(dataset_path, usecols=lambda x: x != "rating_number_historical")

# Check for missing cost column
if 'cost' not in df.columns:
    print('Input is missing required column(s): cost')
    exit()

# Take the test IDs
df_test_ids = pd.read_csv('src/test_record_ids.csv')
df_test = df[df['record_id'].isin(df_test_ids['record_id'])]

# Clean cuisine and type first
df_test['cuisine'] = df_test['cuisine'].str.strip("[] ").str.replace('"', "").str.replace(' ', "")
df_test['type'] = df_test['type'].str.strip("[] ").str.replace('"', "").str.replace(' ', "")
df_test['cuisine_count'] = df_test['cuisine'].apply(lambda x: len(x.split(',')))

X_cols = ['cost', 'cuisine', 'type', 'subzone', 'cuisine_count'] 

# Exclude the target, its derivatives, and identifiers from inputs
X_test = df_test[X_cols]
X_test_processed = preprocessor.transform(X_test)

def predict_func(model, test_df):
    return model.predict(test_df)

y_preds = predict_func(lr_clf, X_test_processed)

# Print preds to a csv file
print("Printing preds to outputs/predictions_clf.csv...\n")
df = pd.DataFrame(y_preds, columns=['Ratings_prediction_clf'])
df.to_csv('outputs/predictions_clf.csv', index=False)

# Test on 3 incomplete cases
# Missing cost value
print("Missing cost value test")
X_test_no_cost_val = X_test.iloc[[0]]
X_test_no_cost_val["cost"] = np.nan
X_test_no_cost_val_processed = preprocessor.transform(X_test_no_cost_val)
print("Prediction: ", predict_func(lr_clf, X_test_no_cost_val_processed))

# Unknown subzone
print("Unknown subzone test")
X_test_unknown_subzone = X_test.iloc[[0]]
X_test_unknown_subzone["subzone"] = "United States of America"
X_test_unknown_subzone_processed = preprocessor.transform(X_test_unknown_subzone)
print("Prediction: ", predict_func(lr_clf, X_test_unknown_subzone_processed))

# Missing cost column
# already checked above





