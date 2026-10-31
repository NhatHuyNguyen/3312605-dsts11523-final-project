# A11 and A12 training pipeline, for CLASSIFICATION problem

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import make_scorer, precision_score
import pandas as pd
import pickle
import copy

from allocate_data import allocate_student_data
dataset_path, allocation = allocate_student_data(student_id="3312605")
MODEL_SEED = allocation["model_seed"]   

print(f"Load and clean data...")
df = pd.read_csv(dataset_path)

# Clean cuisine and type first
df['cuisine'] = df['cuisine'].str.strip("[] ").str.replace('"', "").str.replace(' ', "")
df['type'] = df['type'].str.strip("[] ").str.replace('"', "").str.replace(' ', "")
df['cuisine_count'] = df['cuisine'].apply(lambda x: len(x.split(',')))

# Drop rows without rating
df_with_rating = df.dropna(subset=['rating_number_historical'])

# Create the binary labels for stratification and classification (1 = >= 3.5, 0 = < 3.5)
df_with_rating['class_label'] = (df_with_rating['rating_number_historical'] >= 3.5).astype(int)

# Define the permitted input features
X_cols = ['cost', 'cuisine', 'type', 'subzone', 'lat', 'lng', 'cuisine_count'] 
numeric_features = ['cost', 'cuisine_count', 'lat', 'lng'] 
categorical_features = ['cuisine', 'type', 'subzone']

# Exclude the target, its derivatives, and identifiers from inputs
X = df_with_rating[X_cols]
y_clf = df_with_rating['class_label']
record_ids = df_with_rating['record_id']

# Make 80/20 train/test split, stratified by class label.
# Pass both record_ids and y targets through the split to get the train and test idx
X_train, X_test, y_train_cls, y_test_cls, ids_train, ids_test = train_test_split(
    X, 
    y_clf, 
    record_ids,
    test_size=0.2, 
    random_state=MODEL_SEED, 
    stratify=y_clf  # Stratify using the classification label
)

# Save train and test record ids to file
ids_train.to_csv('./outputs/train_record_ids.csv', index=False)
ids_test.to_csv('./outputs/test_record_ids.csv', index=False)

print(f"Cross validation...\n")
# Classification models
rf_clf = RandomForestClassifier(max_depth=10, n_estimators=100, random_state=MODEL_SEED)
lr_clf = LogisticRegression()
gb_clf = GradientBoostingClassifier(random_state=MODEL_SEED)
base_clf = DummyClassifier(strategy="most_frequent")

models_clf = {
    "baseline": base_clf,
    "rf": rf_clf,
    "lr": lr_clf,
    "grboost": gb_clf
}

# metrics
custom_precision_scorer = make_scorer(precision_score, zero_division=0) # suppress precision warning
scoring_metrics_clf = {
    'acc': 'accuracy',
    'pre': custom_precision_scorer,
    'rec': 'recall',
    'f1': 'f1',
    'f1_macro': 'f1_macro',
}

# Preprocessing pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')), # Fill gaps in numeric features
    ('scaler', StandardScaler())     # Scale numeric features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')), # Fill gaps in categorical data
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Run CV, choose best model
best_model = None
best_score = -9999
best_score_std = 0
for name, model in models_clf.items():
    # Create the full modeling pipeline (Preprocessing + Model)
    preprocessor_copy = copy.deepcopy(preprocessor)
    reg_pipe_each = Pipeline(steps=[
        ('preprocessor', preprocessor_copy),
        ('classifier', model)
    ])
    score = cross_validate(reg_pipe_each, X_train, y_train_cls, cv=5, scoring=scoring_metrics_clf)
    print(f"name: {name}")
    print(f"Accuracy: {score['test_acc'].mean():.4f} +- {score['test_acc'].std():.4f}")
    print(f"Precision: {score['test_pre'].mean():.4f} +- {score['test_pre'].std():.4f}")
    print(f"Recall: {score['test_rec'].mean():.4f} +- {score['test_rec'].std():.4f}")
    print(f"F1: {score['test_f1'].mean():.4f} +- {score['test_f1'].std():.4f}")
    print(f"F1 Macro: {score['test_f1_macro'].mean():.4f} +- {score['test_f1_macro'].std():.4f}\n")
    if score['test_f1_macro'].mean() > best_score:
        best_model = name
        best_score = score['test_f1_macro'].mean()
        best_score_std = score['test_f1_macro'].std()
    # choose lr, based on printed results
    # all models not fitted in CV

print(f"Best model: {best_model} " +
      f"with F1-macro mean score: {best_score:.4f} +- {best_score_std:.4f}\n")


print(f"Fit best model and save model + preprocessor...")
X_train_processed = preprocessor.fit_transform(X_train)
lr_clf = LogisticRegression()
lr_clf.fit(X_train_processed, y_train_cls)

with open('./models/model_clf.pkl', 'wb') as file:
    pickle.dump(lr_clf, file)

with open('./models/preprocess.pkl', 'wb') as file:
    pickle.dump(preprocessor, file)

print(f"Finished training pipeline")






