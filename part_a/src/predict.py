import pickle
import pandas as pd
import numpy as np

# load from allocate_data to test

DATASET_PATH = "./data/allocated/student_restaurants_3312605.csv"

# load saved models
def load_files():
    with open('./models/model_clf.pkl', 'rb') as file:
        lr_clf = pickle.load(file)

    with open('./models/preprocess.pkl', 'rb') as file:
        preprocessor = pickle.load(file)

    return lr_clf, preprocessor



def load_data(preprocessor):
# Make the read test file does NOT have rating, by dropping the rating column
    df = pd.read_csv(DATASET_PATH, usecols=lambda x: x != "rating_number_historical")

    # Check for missing cost column
    if 'cost' not in df.columns:
        print('Input is missing required column(s): cost')
        exit()

    # Take the test IDs
    df_test_ids = pd.read_csv('./outputs/test_record_ids.csv')
    df_test = df[df['record_id'].isin(df_test_ids['record_id'])]

    # Clean cuisine and type first
    df_test['cuisine'] = df_test['cuisine'].str.strip("[] ").str.replace('"', "").str.replace(' ', "")
    df_test['type'] = df_test['type'].str.strip("[] ").str.replace('"', "").str.replace(' ', "")
    df_test['cuisine_count'] = df_test['cuisine'].apply(lambda x: len(x.split(',')))

    X_cols = ['cost', 'cuisine', 'type', 'subzone', 'cuisine_count'] 

    # Exclude the target, its derivatives, and identifiers from inputs
    X_test = df_test[X_cols]
    X_test_processed = preprocessor.transform(X_test)
    return X_test, X_test_processed


def predict_and_save(model, test_df):
    y_preds = model.predict(test_df)

    # Print preds to a csv file
    print("Printing preds to outputs/predictions_clf.csv...\n")
    df = pd.DataFrame(y_preds, columns=['Ratings_prediction_clf'])
    df.to_csv('./outputs/predictions_clf.csv', index=False)


def test_cases(X_test, preprocessor, model):
    # Test on 3 incomplete cases
    # Missing cost value
    print("Missing cost value test")
    X_test_no_cost_val = X_test.iloc[[0]]
    X_test_no_cost_val["cost"] = np.nan
    X_test_no_cost_val_processed = preprocessor.transform(X_test_no_cost_val)
    print("Prediction: ", model.predict(X_test_no_cost_val_processed))

    # Unknown subzone
    print("Unknown subzone test")
    X_test_unknown_subzone = X_test.iloc[[0]]
    X_test_unknown_subzone["subzone"] = "United States of America"
    X_test_unknown_subzone_processed = preprocessor.transform(X_test_unknown_subzone)
    print("Prediction: ", model.predict(X_test_unknown_subzone_processed))

    # Missing cost column
    # already checked above

if __name__ == "__main__":
    model, preprocessor = load_files()
    X_test, X_test_processed = load_data(preprocessor)
    predict_and_save(model, X_test_processed)
    test_cases(X_test, preprocessor, model)



