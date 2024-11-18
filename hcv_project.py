import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

#Uploading and viewing assay dataset
newdata=pd.read_csv('descriptors.csv')
newdata

# Cleaning and Preparing dataset for training and testing
    newdata = newdata.drop(['Name'],axis=1)
    missing_values=newdata.isna().sum()
    newcolumnlabels=list(newdata.columns)
         
    newdata = np.where(np.isinf(newdata), np.nan, newdata)
    newdata=pd.DataFrame(newdata)
    newdata.columns=newcolumnlabels
    newdata['gmin'] = 0
    print(newdata['gmin'].unique() == [0])
                
    imp=SimpleImputer(missing_values=np.nan,strategy='mean')
    newdata1=imp.fit_transform(newdata)
    newdata1=pd.DataFrame(newdata1)
    newdata1

    newdata1.columns=newcolumnlabels
    missing_values=newdata1.isna().sum()
    missing_values
    
    # Scale the data
    scaler = StandardScaler()
    newdata_scaled = scaler.fit_transform(newdata1)
    newdata_scaled

 

activity=pd.read_csv('Daisy.csv')
activity
classlabels=activity.drop(['PUBCHEM_RESULT_TAG','PUBCHEM_EXT_DATASOURCE_SMILES'], axis=1)
classlabels

classlabels= classlabels.dropna() #and then drop rows with missing values
classlabels['PUBCHEM_ACTIVITY_OUTCOME']=(classlabels['PUBCHEM_ACTIVITY_OUTCOME']=='Active').astype(int)
classlabels

activitylabels=list(classlabels.columns)
class_counts = classlabels['PUBCHEM_ACTIVITY_OUTCOME'].value_counts()
print(class_counts)
classlabels['PUBCHEM_ACTIVITY_OUTCOME'].value_counts().plot(kind='bar')

complete=pd.concat([newdata_scaled,classlabels], axis=1)
complete
complete['PUBCHEM_ACTIVITY_OUTCOME'].unique()


train, test = train_test_split(complete, test_size=0.25, random_state=42)
train= train.reset_index(drop=True)
test= test.reset_index(drop=True)
print("Training set shape:", train.shape)
print("Testing set shape:", test.shape)

class_counts = train['PUBCHEM_ACTIVITY_OUTCOME'].value_counts()
print(class_counts)
train['PUBCHEM_ACTIVITY_OUTCOME'].value_counts().plot(kind='bar')

class_counts = test['PUBCHEM_ACTIVITY_OUTCOME'].value_counts()
print(class_counts)
test['PUBCHEM_ACTIVITY_OUTCOME'].value_counts().plot(kind='bar')

x_train=train[train.columns[:-1]].values
y_train=train[train.columns[-1]].values
x_test=test[test.columns[:-1]].values
y_test=test[test.columns[-1]].values

x_train=pd.DataFrame(x_train)
y_train=pd.DataFrame(y_train)
x_test=pd.DataFrame(x_test)
y_test=pd.DataFrame(y_test)

smote = SMOTE(random_state=42)
# SMOTE # RandomOversampler generates duplicates to match your classes.
x_train_resampled, y_train_resampled= smote.fit_resample(x_train, y_train)

y_train_resampled.columns=activitylabels
y_test.columns=activitylabels

x_test.to_csv('x_test.csv', index=False, header=True)
x_train_resampled.to_csv('x_train_resampled.csv',index=False,header=True)
y_test.to_csv('y_test.csv', index=False, header=True)
y_train_resampled.to_csv('y_train_resampled.csv',index=False,header=True)

class_counts = y_train_resampled['PUBCHEM_ACTIVITY_OUTCOME'].value_counts()
print(class_counts)
y_train_resampled['PUBCHEM_ACTIVITY_OUTCOME'].value_counts().plot(kind='bar')

from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, f1_score, recall_score
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

normalize = MinMaxScaler()
normalize.fit(x_train_resampled)
# Transform both training and testing data
added_normalized_train = normalize.transform(x_train_resampled)
added_normalized_test = normalize.transform(x_test)

high_variance_data = PCA(n_components=0.9,random_state=42)
high_variance_data.fit(added_normalized_train)
x_train_selected = high_variance_data.transform(added_normalized_train)
x_test_selected = high_variance_data.transform(added_normalized_test)
x_train_selected = pd.DataFrame(x_train_selected)
x_test_selected = pd.DataFrame(x_test_selected)

x_train_selected

x_test_selected.to_csv('x_test_selected.csv', index=False, header=True)
x_train_selected.to_csv('x_train_selected.csv',index=False,header=True)

#Model Training and Testing

# Train and evaluate SVM model with best hyperparameters
best_svm = SVC(**{'C': 10, 'kernel': 'linear'},random_state=42)
best_svm.fit(x_train_selected, y_train_resampled)
prediction_svm = best_svm.predict(x_test_selected)
report_svm = classification_report(y_test, prediction_svm)
print(report_svm)

# Define the scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted')
}

# Perform k-fold cross-validation
results = cross_validate(
    estimator= best_svm,  # replace with your model
    X=x_train_selected,  # replace with your data
    y=y_train_resampled,  # replace with your target variable
    cv=5,  # number of folds
    scoring=scoring
)

# Print the results
print("Accuracy:", results['test_accuracy'])
print("Precision:", results['test_precision'])
print("F1 score:", results['test_f1'])
print("Recall:", results['test_recall'])

from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt

best_svm = SVC(**{'C': 10, 'kernel': 'linear'},random_state=42, probability=True)
best_svm.fit(x_train_selected, y_train_resampled)
y_scores = best_svm.predict_proba(x_test_selected)[:, 1] #replace model with your model

# Calculate the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Calculate the Area Under the ROC Curve (AUC)
auc_score = roc_auc_score(y_test, y_scores)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for SVM Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve_for_SVM.png', dpi=300, bbox_inches='tight')
plt.show()

from sklearn.ensemble import RandomForestClassifier

#rfc initialization with 100 number of estimators
best_rfc = RandomForestClassifier(**{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 300})
#lets fit the training data and make predictions and evaluate the performance
best_rfc.fit(x_train_selected, y_train_resampled)
# Make predictions on the selected features
prediction_rfc =best_rfc.predict(x_test_selected)
report_rfc = classification_report(y_test, prediction_rfc)
print(report_rfc)

# Define the scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted'),
    'f1': make_scorer(f1_score, average='weighted'),
    'recall': make_scorer(recall_score, average='weighted')
}

# Perform k-fold cross-validation
results = cross_validate(
    estimator=best_rfc,  # replace with your model
    X=x_train_selected,  # replace with your data
    y=y_train_resampled,  # replace with your target variable
    cv=5,  # number of folds
    scoring=scoring
)

# Print the results
print("Accuracy:", results['test_accuracy'])
print("Precision:", results['test_precision'])
print("F1 score:", results['test_f1'])
print("Recall:", results['test_recall'])

#rfc initialization with 100 number of estimators
y_scores = best_rfc.predict_proba(x_test_selected)[:, 1] #replace model with your model

# Calculate the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Calculate the Area Under the ROC Curve (AUC)
auc_score = roc_auc_score(y_test, y_scores)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for RF Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve_for_SVM.png', dpi=300, bbox_inches='tight')
plt.show()

mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam') #lets initialize the mlp classifier
mlp.fit(x_train_selected,y_train_resampled)
prediction_mlp = mlp.predict(x_test_selected)
report_mlp = classification_report(y_test, prediction_mlp)
print(report_mlp)

from sklearn.model_selection import KFold

# Define K-fold cross-validation
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize a list to store the reports
reports = []

# Perform K-fold cross-validation
for train_index, test_index in kfold.split(x_train_resampled):
    X_train_fold, X_test_fold = x_train_selected.iloc[train_index], x_train_selected.iloc[test_index]
    y_train_fold, y_test_fold = y_train_resampled.iloc[train_index],  y_train_resampled.iloc[test_index]

    # Train the MLP classifier on the current fold
    mlp.fit(X_train_fold, y_train_fold)

    # Make predictions on the test set
    predictions = mlp.predict(X_test_fold)

    # Generate the classification report
    report = classification_report(y_test_fold, predictions)

    # Append the report to the list
    reports.append(report)

# Print the average report across all folds
print("Average Report:")
for report in reports:
    print(report)

y_scores = mlp.predict_proba(x_test_selected)[:, 1] #replace model with your model

# Calculate the false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_scores)

# Calculate the Area Under the ROC Curve (AUC)
auc_score = roc_auc_score(y_test, y_scores)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line for random guessing
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve for MLP Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('roc_curve_for_SVM.png', dpi=300, bbox_inches='tight')
plt.show()

query=pd.read_csv('test_descriptors.csv')

query

query = query.dropna() #and then drop rows with missing values

query

query_impute=imp.transform(query)

query_scaled=scaler.transform(query_impute)

new_queries_scaled_normalised = normalize.transform(query_scaled)
new_queries_selected = high_variance_data.transform(new_queries_scaled_normalised)

new_queries_selected
new_queries_selected=pd.DataFrame(new_queries_selected)

new_queries_selected

new_queries_selected.to_csv('new_queries_selected.csv', index=False, header=True)

Query1= best_svm.predict(new_queries_selected)
Query2= best_rfc.predict(new_queries_selected)
Query3= mlp.predict(new_queries_selected)

Query1

Query2

Query3

import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

x_AD=pd.read_csv('x_AD.csv')

y_AD=pd.read_csv('y_AD.csv')

y_AD

# Linear Regression Model


ols_model = sm.OLS(y_AD, x_AD).fit()

# Extract results
residuals = ols_model.resid
fitted_values = ols_model.fittedvalues
stand_resids = ols_model.resid_pearson
influence = ols_model.get_influence()
leverage = influence.hat_matrix_diag


train_residuals = pd.Series(stand_resids).head(1648)
train_leverage = pd.Series(leverage).head(1648)
query_residuals = pd.Series(stand_resids).tail(4)
query_leverage = pd.Series(leverage).tail(4)

# Data Preparation
# Data Preparation
def prepare_data(residuals, leverage):
    data = pd.DataFrame({
        'Standardized Residual': residuals,
        'Leverage': leverage
    })
    data = data.sample(frac=1)  # Shuffle data
    return data

train_data = prepare_data(train_residuals, train_leverage)
query_data = prepare_data(query_residuals, query_leverage)


# Define Applicability domain
leverage_limit = 0.3
residual_limit = 2

# Identify compounds within applicability domain
train_data['Applicability Domain'] = np.where((train_data['Leverage'] <= leverage_limit) &
                                         (np.abs(train_data['Standardized Residual']) <= residual_limit),
                                         'Within Domain', 'Outside Domain')
query_data['Applicability Domain'] = np.where((query_data['Leverage'] <= leverage_limit) &
                                         (np.abs(query_data['Standardized Residual']) <= residual_limit),
                                         'Within Domain', 'Outside Domain')
train_data['activity'] = 0
query_data['activity'] = 1

# Visualization
sns.set(font_scale=1.4)
plt.figure(figsize=(15, 8))
sns.scatterplot(data=train_data, x="Leverage", y="Standardized Residual",
                hue="activity", s=120,palette=['orange','blue'])
sns.scatterplot(data=query_data, x="Leverage", y="Standardized Residual",
                hue="activity", s=120)

# Add horizontal and vertical lines
plt.axhline(y=residual_limit, ls='--', c='grey', label='Upper Residual Limit')
plt.axhline(y=-residual_limit, ls='--', c='grey', label='Lower Residual Limit')
plt.axvline(x=leverage_limit, ls='--', c='grey', label='Leverage Limit')
plt.legend()
plt.title("Williams plot of standardized residual versus leverage")
plt.show()
plt.savefig("plot.png")

# Print summary statistics
print("Compounds within applicability domain:", len(train_data[train_data['Applicability Domain'] == 'Within Domain']))
print("Compounds outside applicability domain:", len(train_data[train_data['Applicability Domain'] == 'Outside Domain']))

L_selected=pd.read_csv('L_selected.csv')
S_selected=pd.read_csv('S_selected.csv')
D_selected=pd.read_csv('D_selected.csv')
Q_selected=pd.read_csv('Q_selected.csv')

# Predict probabilities for a query
query_probabilities = mlp.predict_proba(L_selected)

# Calculate confidence score as maximum probability
confidence_score = np.max(query_probabilities)

print("Confidence Score:", confidence_score)

# Predict probabilities for a query
query_probabilities = mlp.predict_proba(S_selected)

# Calculate confidence score as maximum probability
confidence_score = np.max(query_probabilities)

print("Confidence Score:", confidence_score)

# Predict probabilities for a query
query_probabilities = mlp.predict_proba(D_selected)

# Calculate confidence score as maximum probability
confidence_score = np.max(query_probabilities)

print("Confidence Score:", confidence_score)

# Predict probabilities for a query
query_probabilities = mlp.predict_proba(Q_selected)

# Calculate confidence score as maximum probability
confidence_score = np.max(query_probabilities)

print("Confidence Score:", confidence_score)
