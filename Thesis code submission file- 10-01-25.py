#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
import warnings
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, roc_curve, auc
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_val_score


df = pd.read_csv('Cardiovascular_Disease_Dataset.csv')
df.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


# separating and splitting of data


# In[22]:


X = df.drop('target', axis=1)  
y = df['target']  


# In[23]:


X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)


# In[31]:


X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")


# # EDA: descriptive statistics, correlation and feature importance

# In[272]:


continuous_features = ['age', 'restingBP', 'serumcholestrol', 'maxheartrate', 'oldpeak']
categorical_features = ['gender', 'chestpain', 'fastingbloodsugar', 'restingrelectro', 
                        'exerciseangia', 'slope', 'noofmajorvessels']

fig, axes = plt.subplots(len(continuous_features), 2, figsize=(14, 5 * len(continuous_features)))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

for i, feature in enumerate(continuous_features):
    sns.histplot(X_train[feature], kde=True, bins=30, color='blue', ax=axes[i, 0])  
    axes[i, 0].set_title(f'Distribution of {feature}')
    axes[i, 0].set_xlabel(feature)
    axes[i, 0].set_ylabel('Frequency')
    axes[i, 0].grid(True)

    sns.boxplot(x=X_train[feature], color='green', ax=axes[i, 1])  
    axes[i, 1].set_title(f'Boxplot of {feature}')
    axes[i, 1].set_xlabel(feature)

plt.savefig('continuous_features_distribution_2columns_train.png')

fig2, axes2 = plt.subplots((len(categorical_features) + 1) // 2, 2, figsize=(14, 5 * ((len(categorical_features) + 1) // 2)))
plt.subplots_adjust(hspace=0.5, wspace=0.3)

for i, feature in enumerate(categorical_features):
    row = i // 2
    col = i % 2
    sns.countplot(x=X_train[feature], palette='viridis', ax=axes2[row, col])  
    axes2[row, col].set_title(f'Distribution of {feature}')
    axes2[row, col].set_xlabel(feature)
    axes2[row, col].set_ylabel('Count')
    axes2[row, col].grid(True)

if len(categorical_features) % 2 != 0:
    axes2[-1, -1].axis('off')

plt.savefig('categorical_features_distribution_2columns_train.png')

continuous_summary = X_train[continuous_features].describe().T  # Summary statistics for continuous features
categorical_summary = X_train[categorical_features].apply(lambda x: x.value_counts()).T  # Counts for categorical features

print("Continuous Features Summary:")
print(continuous_summary)

print("\nCategorical Features Summary:")
print(categorical_summary)


# In[286]:


train_data = X_train.copy()
train_data['target'] = y_train

correlation_matrix = train_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.savefig('heatmap.png')
plt.show()
print(correlation_matrix)


# In[281]:


rfmodel = RandomForestClassifier()
rfmodel.fit(X_train, y_train)

importances = rfmodel.feature_importances_

feature_importances = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print(feature_importances)

plt.figure(figsize=(10, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance for Heart Disease Prediction', fontsize=16)
#plt.gca().invert_yaxis() 
plt.tight_layout()

plt.savefig('feature_importance.png', dpi=300)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # feature selection using sequential feature selector on cv5

# In[136]:


for num_features in range(4, 13):
    print(f"\nselecting {num_features} features")
    
    sfs = SequentialFeatureSelector(model, n_features_to_select=num_features, direction='forward', cv=5)
    
    sfs.fit(X_train, y_train)
    selected_features = sfs.get_support()    
    print(f"selected features: {selected_features}")
    
    X_selected = sfs.transform(X_train)
    
    cv_scores = cross_val_score(model, X_selected, y_train, cv=5, scoring='accuracy')  
    print(f"cv scores with {num_features} features: {cv_scores}")
    print(f"mean cv score: {cv_scores.mean()}")


# In[35]:


sfs = SequentialFeatureSelector(model, n_features_to_select=5, direction='forward', cv=5)
sfs.fit(X_train, y_train)
selected_features = sfs.get_support()    
print(f"selected features: {selected_features}")


# In[36]:


X_train_selected = X_train.loc[:, selected_features]
X_val_selected = X_val.loc[:, selected_features]
X_test_selected = X_test.loc[:, selected_features]


# In[39]:


print(f"Training set size: {X_train_selected.shape[0]}")
print(f"Validation set size: {X_val_selected.shape[0]}")
print(f"Test set size: {X_test_selected.shape[0]}")


# In[135]:


X_train_selected


# In[ ]:





# In[ ]:





# # training and tuning black box models

# In[130]:


from sklearn.model_selection import RandomizedSearchCV, train_test_split, KFold
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, make_scorer
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np



rf_param_dist3 = {
    'n_estimators': [100, 150, 200, 300],  # Reduce the number of trees for efficiency and reduced complexity.
    'max_depth': [3,4],  # Limit tree depth to reduce the risk of overfitting.
    'min_samples_split': [10, 20],  # Higher minimum samples for splits to avoid overly specific splits.
    'min_samples_leaf': [4, 6],  # Increase minimum samples per leaf to limit tree depth indirectly.
    'max_features': ['sqrt', 'log2'],  # Use a subset of features to enhance generalization.
    'bootstrap': [True, False]  # Test both bootstrapped and full sample trees.
}


xgb_param_dist = {
    'n_estimators': [100, 200],  # Limit the number of boosting rounds to prevent overfitting.
    'learning_rate': [0.01, 0.05],  # Stick to smaller learning rates for better generalization.
    'max_depth': [3, 4],  # Lower maximum depth to prevent overly complex trees.
    'subsample': [0.5, 0.7, 0.8],  # Focus on smaller subsets to encourage diversity in trees.
    'colsample_bytree': [0.5, 0.7],  # Reduce the fraction of features used for building trees.
    'reg_alpha': [0, 1, 5],  # Add L1 regularization to limit overfitting.
    'reg_lambda': [1, 5, 10],  # Add L2 regularization for stronger constraints.
    'min_child_weight': [1, 5, 10]  # Ensure splits require more significant data points.
}


scoring = make_scorer(f1_score)


xgb_model_2 = XGBClassifier(use_label_encoder=False, eval_metric='logloss', tree_method='auto')

xgb_random_search_2 = RandomizedSearchCV(xgb_model_2, xgb_param_dist, n_iter=50, scoring=scoring, cv=5, verbose=1, 
                                         random_state=42, n_jobs=-1)

xgb_random_search_2.fit(X_train_selected, y_train)

xgb_best_2 = xgb_random_search_2.best_estimator_

xgb_train_pred = xgb_best_2.predict(X_train_selected)
xgb_train_pred_proba = xgb_best_2.predict_proba(X_train_selected)

xgb_train_accuracy = accuracy_score(y_train, xgb_train_pred)
xgb_train_log_loss = log_loss(y_train, xgb_train_pred_proba)
xgb_train_precision = precision_score(y_train, xgb_train_pred)
xgb_train_recall = recall_score(y_train, xgb_train_pred)
xgb_train_f1_score = f1_score(y_train, xgb_train_pred)

xgb_val_pred = xgb_best_2.predict(X_val_selected)
xgb_val_pred_proba = xgb_best_2.predict_proba(X_val_selected)

xgb_val_accuracy = accuracy_score(y_val, xgb_val_pred)
xgb_val_log_loss = log_loss(y_val, xgb_val_pred_proba)
xgb_val_precision = precision_score(y_val, xgb_val_pred)
xgb_val_recall = recall_score(y_val, xgb_val_pred)
xgb_val_f1_score = f1_score(y_val, xgb_val_pred)

xgb_y_pred_2 = xgb_best_2.predict(X_test_selected)
xgb_y_pred_proba_2 = xgb_best_2.predict_proba(X_test_selected)

xgb_test_accuracy = accuracy_score(y_test, xgb_y_pred_2)
xgb_test_log_loss = log_loss(y_test, xgb_y_pred_proba_2)
xgb_test_precision = precision_score(y_test, xgb_y_pred_2)
xgb_test_recall = recall_score(y_test, xgb_y_pred_2)
xgb_test_f1_score = f1_score(y_test, xgb_y_pred_2)

print("\nXGBoost Train Performance:")
print(f"Accuracy: {xgb_train_accuracy:.4f}")
print(f"Log Loss: {xgb_train_log_loss:.4f}")
print(f"Precision: {xgb_train_precision:.4f}")
print(f"Recall: {xgb_train_recall:.4f}")
print(f"F1 Score: {xgb_train_f1_score:.4f}")

print("\nXGBoost Validation Performance:")
print(f"Accuracy: {xgb_val_accuracy:.4f}")
print(f"Log Loss: {xgb_val_log_loss:.4f}")
print(f"Precision: {xgb_val_precision:.4f}")
print(f"Recall: {xgb_val_recall:.4f}")
print(f"F1 Score: {xgb_val_f1_score:.4f}")

print("\nXGBoost Test Performance:")
print(f"Accuracy: {xgb_test_accuracy:.4f}")
print(f"Log Loss: {xgb_test_log_loss:.4f}")
print(f"Precision: {xgb_test_precision:.4f}")
print(f"Recall: {xgb_test_recall:.4f}")
print(f"F1 Score: {xgb_test_f1_score:.4f}")

print(xgb_random_search_2.best_params_)


rf_model_2 = RandomForestClassifier(random_state=42)

rf_random_search_2 = RandomizedSearchCV(rf_model_2, rf_param_dist3, n_iter=50, scoring=scoring, cv=5, verbose=1, 
                                        random_state=42, n_jobs=-1)

rf_random_search_2.fit(X_train_selected, y_train)

rf_best_2 = rf_random_search_2.best_estimator_

rf_train_pred = rf_best_2.predict(X_train_selected)
rf_train_pred_proba = rf_best_2.predict_proba(X_train_selected)

rf_train_accuracy = accuracy_score(y_train, rf_train_pred)
rf_train_log_loss = log_loss(y_train, rf_train_pred_proba)
rf_train_precision = precision_score(y_train, rf_train_pred)
rf_train_recall = recall_score(y_train, rf_train_pred)
rf_train_f1_score = f1_score(y_train, rf_train_pred)

rf_val_pred = rf_best_2.predict(X_val_selected)
rf_val_pred_proba = rf_best_2.predict_proba(X_val_selected)

rf_val_accuracy = accuracy_score(y_val, rf_val_pred)
rf_val_log_loss = log_loss(y_val, rf_val_pred_proba)
rf_val_precision = precision_score(y_val, rf_val_pred)
rf_val_recall = recall_score(y_val, rf_val_pred)
rf_val_f1_score = f1_score(y_val, rf_val_pred)

rf_y_pred_2 = rf_best_2.predict(X_test_selected)
rf_y_pred_proba_2 = rf_best_2.predict_proba(X_test_selected)

rf_test_accuracy = accuracy_score(y_test, rf_y_pred_2)
rf_test_log_loss = log_loss(y_test, rf_y_pred_proba_2)
rf_test_precision = precision_score(y_test, rf_y_pred_2)
rf_test_recall = recall_score(y_test, rf_y_pred_2)
rf_test_f1_score = f1_score(y_test, rf_y_pred_2)

print("\nRandom Forest Train Performance:")
print(f"Accuracy: {rf_train_accuracy:.4f}")
print(f"Log Loss: {rf_train_log_loss:.4f}")
print(f"Precision: {rf_train_precision:.4f}")
print(f"Recall: {rf_train_recall:.4f}")
print(f"F1 Score: {rf_train_f1_score:.4f}")

print("\nRandom Forest Validation Performance:")
print(f"Accuracy: {rf_val_accuracy:.4f}")
print(f"Log Loss: {rf_val_log_loss:.4f}")
print(f"Precision: {rf_val_precision:.4f}")
print(f"Recall: {rf_val_recall:.4f}")
print(f"F1 Score: {rf_val_f1_score:.4f}")

print("\nRandom Forest Test Performance:")
print(f"Accuracy: {rf_test_accuracy:.4f}")
print(f"Log Loss: {rf_test_log_loss:.4f}")
print(f"Precision: {rf_test_precision:.4f}")
print(f"Recall: {rf_test_recall:.4f}")
print(f"F1 Score: {rf_test_f1_score:.4f}")

print(rf_random_search_2.best_params_)


# # distillation

# In[355]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def distill_softlabels2(model_teacher, X_train, X_val, X_test, y_train, y_val, y_test):
    # first generate soft labels (probabilities) from the teacher model
    soft_labels_train = model_teacher.predict_proba(X_train)[:, 1]
    soft_labels_val = model_teacher.predict_proba(X_val)[:, 1]
    
    # then train the student model (Decision Tree)
    model_student = DecisionTreeRegressor(random_state=1234, max_depth=4)
    model_student.fit(X_train, soft_labels_train)
    
    #  predict on training, validation, and test sets
    y_pred_train_prob = model_student.predict(X_train)
    y_pred_val_prob = model_student.predict(X_val)
    y_pred_test_prob = model_student.predict(X_test)
    
    # convert probabilities to binary predictions
    y_pred_train = (y_pred_train_prob >= 0.5).astype(int)
    y_pred_val = (y_pred_val_prob >= 0.5).astype(int)
    y_pred_test = (y_pred_test_prob >= 0.5).astype(int)
    
    # compute metrics for training set
    acc_train = accuracy_score(y_train, y_pred_train)
    prec_train = precision_score(y_train, y_pred_train)
    recall_train = recall_score(y_train, y_pred_train)
    f1_train = f1_score(y_train, y_pred_train)
    logloss_train = log_loss(y_train, y_pred_train_prob)
    
    # compute metrics for validation set
    acc_val = accuracy_score(y_val, y_pred_val)
    prec_val = precision_score(y_val, y_pred_val)
    recall_val = recall_score(y_val, y_pred_val)
    f1_val = f1_score(y_val, y_pred_val)
    logloss_val = log_loss(y_val, y_pred_val_prob)
    
    # compute metrics for test set
    acc_test = accuracy_score(y_test, y_pred_test)
    prec_test = precision_score(y_test, y_pred_test)
    recall_test = recall_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test)
    logloss_test = log_loss(y_test, y_pred_test_prob)
    
    print("\nTraining Set Performance:")
    print(f"accuracy: {acc_train:.4f}, logloss: {logloss_train:.4f}, precision: {prec_train:.4f}, recall: {recall_train:.4f}, F1 score: {f1_train:.4f}")
    
    print("\nValidation Set Performance:")
    print(f"accuracy: {acc_val:.4f}, logloss: {logloss_val:.4f}, precision: {prec_val:.4f}, recall: {recall_val:.4f}, F1 score: {f1_val:.4f}")
    
    print("\nTest Set Performance:")
    print(f"accuracy: {acc_test:.4f}, logloss: {logloss_test:.4f}, precision: {prec_test:.4f}, recall: {recall_test:.4f}, F1 score: {f1_test:.4f}")
    
    # ROC Curve for test set
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_test_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve and AUC value on test set')
    plt.legend()
    plt.show()

    return model_student, y_pred_test


# In[356]:


modelA_soft, y_pred_A = distill_softlabels2(xgb_best_2, X_train_selected, X_val_selected, X_test_selected, y_train, y_val, y_test)


# In[357]:


modelB_soft, y_pred_B = distill_softlabels2(rf_best_2, X_train_selected, X_val_selected, X_test_selected, y_train, y_val, y_test)


# # analysis of depth in distilled models

# In[202]:


depths = [1, 2, 3, 4, 5, 6, 7, 8]
metrics_A = {
    "accuracy": [0.9250, 0.9250, 0.9450, 0.9500, 0.9450, 0.9400, 0.9500, 0.9500],
    "logloss": [0.2420, 0.1949, 0.1736, 0.1429, 0.1555, 0.1429, 0.1661, 0.1487],
    "precision": [0.9811, 0.9811, 0.9569, 0.9496, 0.9492, 0.9339, 0.9573, 0.9573],
    "recall": [0.8889, 0.8889, 0.9487, 0.9658, 0.9573, 0.9658, 0.9573, 0.9573],
    "f1_score": [0.9327, 0.9327, 0.9528, 0.9576, 0.9532, 0.9496, 0.9573, 0.9573],
}
metrics_B = {
    "accuracy": [0.9250, 0.9250, 0.9250, 0.9550, 0.9450, 0.9500, 0.9500, 0.9500],
    "logloss": [0.2478, 0.2003, 0.2003, 0.1370, 0.1276, 0.1237, 0.1236, 0.1214],
    "precision": [0.9811, 0.9811, 0.9811, 0.9821, 0.9649, 0.9652, 0.9652, 0.9652],
    "recall": [0.8889, 0.8889, 0.8889, 0.9402, 0.9402, 0.9487, 0.9487, 0.9487],
    "f1_score": [0.9327, 0.9327, 0.9327, 0.9607, 0.9524, 0.9569, 0.9569, 0.9569],
}

df_A = pd.DataFrame(metrics_A, index=depths)
df_B = pd.DataFrame(metrics_B, index=depths)

fig, axes = plt.subplots(3, 2, figsize=(12, 10))
fig.suptitle("Performance Metrics Across Depths", fontsize=16)

metrics = ["accuracy", "logloss", "precision", "recall", "f1_score"]
for ax, metric in zip(axes.flatten(), metrics):
    ax.plot(depths, df_A[metric], label="Model A", marker="o")
    ax.plot(depths, df_B[metric], label="Model B", marker="o")
    ax.set_title(metric.capitalize())
    ax.set_xlabel("Depth")
    ax.set_ylabel(metric.capitalize())
    ax.legend()
    ax.grid(True)

axes[-1, -1].axis("off")

plt.tight_layout(rect=[0, 0, 1, 0.96])
output_path = "performance_metrics_plot for A and B.png"
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Plot saved as {output_path}")


# In[207]:


depths = [1, 2, 3, 4, 5, 6, 7, 8]
metrics_A = {
    "accuracy": [0.9250, 0.9250, 0.9450, 0.9500, 0.9450, 0.9400, 0.9500, 0.9500],
    "precision": [0.9811, 0.9811, 0.9569, 0.9496, 0.9492, 0.9339, 0.9573, 0.9573],
    "recall": [0.8889, 0.8889, 0.9487, 0.9658, 0.9573, 0.9658, 0.9573, 0.9573],
}
metrics_B = {
    "accuracy": [0.9250, 0.9250, 0.9250, 0.9550, 0.9450, 0.9500, 0.9500, 0.9500],
    "precision": [0.9811, 0.9811, 0.9811, 0.9821, 0.9649, 0.9652, 0.9652, 0.9652],
    "recall": [0.8889, 0.8889, 0.8889, 0.9402, 0.9402, 0.9487, 0.9487, 0.9487],
}

df_A = pd.DataFrame(metrics_A, index=depths)
df_B = pd.DataFrame(metrics_B, index=depths)

def plot_metrics(df, model_name, output_path):
    plt.figure(figsize=(10, 6))
    for metric in df.columns:
        plt.plot(depths, df[metric], marker="o", label=metric.capitalize())
    plt.title(f"Performance Metrics for {model_name}", fontsize=14)
    plt.xlabel("Depth", fontsize=12)
    plt.ylabel("Metric Values", fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend(title="Metrics", fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.show()

plot_metrics(df_A, "Model A", "model_a_metrics.png")

plot_metrics(df_B, "Model B", "model_b_metrics.png")

print("Plots saved as 'model_a_metrics.png' and 'model_b_metrics.png'")


# In[ ]:





# # error analysis

# In[142]:


from sklearn.metrics import confusion_matrix, classification_report


results = pd.DataFrame({
    'True': y_test,
    'Predicted_XGB': xgb_y_pred_2,
    'Predicted_RF': rf_y_pred_2
})

results['XGB_Error'] = results['True'] != results['Predicted_XGB']
results['RF_Error'] = results['True'] != results['Predicted_RF']

confusion_xgb = confusion_matrix(y_test, xgb_y_pred_2)
confusion_rf = confusion_matrix(y_test, rf_y_pred_2)

tn_xgb, fp_xgb, fn_xgb, tp_xgb = confusion_xgb.ravel()

tn_rf, fp_rf, fn_rf, tp_rf = confusion_rf.ravel()

print("XGBoost Errors:")
print(f"True Positives (TP): {tp_xgb}")
print(f"True Negatives (TN): {tn_xgb}")
print(f"False Positives (FP): {fp_xgb}")
print(f"False Negatives (FN): {fn_xgb}")

print("\nRandom Forest Errors:")
print(f"True Positives (TP): {tp_rf}")
print(f"True Negatives (TN): {tn_rf}")
print(f"False Positives (FP): {fp_rf}")
print(f"False Negatives (FN): {fn_rf}")


print("XGBoost Classification Report:")
print(classification_report(y_test, xgb_y_pred_2))

print("Random Forest Classification Report:")
print(classification_report(y_test, rf_y_pred_2))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_xgb, annot=True, fmt='d', ax=axes[0], cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
axes[0].set_title('XGBoost Confusion Matrix')

sns.heatmap(confusion_rf, annot=True, fmt='d', ax=axes[1], cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
axes[1].set_title('Random Forest Confusion Matrix')

plt.show()

misclassified_xgb = results[results['XGB_Error']]
misclassified_rf = results[results['RF_Error']]

print("Misclassified Instances (XGBoost):")
print(misclassified_xgb)

print("\nMisclassified Instances (Random Forest):")
print(misclassified_rf)



# In[ ]:





# In[143]:


results = pd.DataFrame({
    'True': y_test,
    'Predicted_modelA': y_pred_A,
    'Predicted_modelB': y_pred_B
})

results['modelA_Error'] = results['True'] != results['Predicted_modelA']
results['modelB_Error'] = results['True'] != results['Predicted_modelB']

confusion_A = confusion_matrix(y_test, y_pred_A)
confusion_B = confusion_matrix(y_test, y_pred_B)

tn_A, fp_A, fn_A, tp_A = confusion_A.ravel()
tn_B, fp_B, fn_B, tp_B = confusion_B.ravel()

print("Model A Errors:")
print(f"True Positives (TP): {tp_A}")
print(f"True Negatives (TN): {tn_A}")
print(f"False Positives (FP): {fp_A}")
print(f"False Negatives (FN): {fn_A}")

print("\nModel B Errors:")
print(f"True Positives (TP): {tp_B}")
print(f"True Negatives (TN): {tn_B}")
print(f"False Positives (FP): {fp_B}")
print(f"False Negatives (FN): {fn_B}")

print("modelA Classification Report:")
print(classification_report(y_test, y_pred_A))

print("modelB Classification Report:")
print(classification_report(y_test, y_pred_B))

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(confusion_A, annot=True, fmt='d', ax=axes[0], cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
axes[0].set_title('modelA Confusion Matrix')

sns.heatmap(confusion_B, annot=True, fmt='d', ax=axes[1], cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
axes[1].set_title('modelB Confusion Matrix')

plt.show()

misclassified_A = results[results['modelA_Error']]
misclassified_B = results[results['modelB_Error']]

print("Misclassified Instances (modelA):")
print(misclassified_A)

print("\nMisclassified Instances (modelB):")
print(misclassified_B)


# # disparate impact black box models

# ## DIR for gender

# In[137]:


from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results = pd.DataFrame({
    'True': y_test,
    'Predicted_XGB': xgb_y_pred_2,
    'Predicted_RF': rf_y_pred_2
})

results['Gender'] = X_test['gender']  

results_male = results[results['Gender'] == 0]  
results_female = results[results['Gender'] == 1]  

def favorable_outcomes(tp, tn, total):
    return (tp + tn) / total

confusion_xgb_male = confusion_matrix(results_male['True'], results_male['Predicted_XGB'])
tn_xgb_m, fp_xgb_m, fn_xgb_m, tp_xgb_m = confusion_xgb_male.ravel()
prop_xgb_male = favorable_outcomes(tp_xgb_m, tn_xgb_m, len(results_male))

confusion_rf_male = confusion_matrix(results_male['True'], results_male['Predicted_RF'])
tn_rf_m, fp_rf_m, fn_rf_m, tp_rf_m = confusion_rf_male.ravel()
prop_rf_male = favorable_outcomes(tp_rf_m, tn_rf_m, len(results_male))

confusion_xgb_female = confusion_matrix(results_female['True'], results_female['Predicted_XGB'])
tn_xgb_f, fp_xgb_f, fn_xgb_f, tp_xgb_f = confusion_xgb_female.ravel()
prop_xgb_female = favorable_outcomes(tp_xgb_f, tn_xgb_f, len(results_female))

confusion_rf_female = confusion_matrix(results_female['True'], results_female['Predicted_RF'])
tn_rf_f, fp_rf_f, fn_rf_f, tp_rf_f = confusion_rf_female.ravel()
prop_rf_female = favorable_outcomes(tp_rf_f, tn_rf_f, len(results_female))

def disparate_impact_ratio(prop_male, prop_female):
    return prop_male / prop_female if prop_female != 0 else float('nan')

dir_xgb = disparate_impact_ratio(prop_xgb_male, prop_xgb_female)
dir_rf = disparate_impact_ratio(prop_rf_male, prop_rf_female)

print("Proportion of Favorable Outcomes:")
print(f"XGBoost - Male: {prop_xgb_male:.4f}, Female: {prop_xgb_female:.4f}")
print(f"Random Forest - Male: {prop_rf_male:.4f}, Female: {prop_rf_female:.4f}")
print("\nDisparate Impact Ratio (DIR):")
print(f"XGBoost DIR (Male/Female): {dir_xgb:.4f}")
print(f"Random Forest DIR (Male/Female): {dir_rf:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

sns.heatmap(confusion_xgb_male, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
axes[0, 0].set_title('XGBoost Confusion Matrix - Male')

sns.heatmap(confusion_xgb_female, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
axes[0, 1].set_title('XGBoost Confusion Matrix - Female')

sns.heatmap(confusion_rf_male, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
axes[1, 0].set_title('Random Forest Confusion Matrix - Male')

sns.heatmap(confusion_rf_female, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
axes[1, 1].set_title('Random Forest Confusion Matrix - Female')

plt.tight_layout()
plt.show()

print("\nXGBoost Classification Report - Male:")
print(classification_report(results_male['True'], results_male['Predicted_XGB']))

print("\nRandom Forest Classification Report - Male:")
print(classification_report(results_male['True'], results_male['Predicted_RF']))

print("\nXGBoost Classification Report - Female:")
print(classification_report(results_female['True'], results_female['Predicted_XGB']))

print("\nRandom Forest Classification Report - Female:")
print(classification_report(results_female['True'], results_female['Predicted_RF']))


# ## DIR for chestpain

# In[139]:


from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results = pd.DataFrame({
    'True': y_test,
    'Predicted_XGB': xgb_y_pred_2,
    'Predicted_RF': rf_y_pred_2
})

results['ChestPain'] = X_test['chestpain']  

results_cp = {level: results[results['ChestPain'] == level] for level in [0, 1, 2, 3]}

def calculate_confusion_matrix_components(true_labels, predictions):
    confusion = confusion_matrix(true_labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = 0, 0, 0, 0
    if confusion.shape == (2, 2):
        tn, fp, fn, tp = confusion.ravel()
    elif confusion.shape == (1, 2):  
        tn, fp = confusion[0, :]
    elif confusion.shape == (2, 1):  
        fn, tp = confusion[:, 0]
    return tn, fp, fn, tp

def favorable_outcomes(tp, tn, total):
    return (tp + tn) / total if total > 0 else 0

proportions = {}
for model in ['Predicted_XGB', 'Predicted_RF']:
    proportions[model] = {}
    for level, group in results_cp.items():
        tn, fp, fn, tp = calculate_confusion_matrix_components(group['True'], group[model])
        proportions[model][level] = favorable_outcomes(tp, tn, len(group))

dirs = {
    model: {
        level: proportions[model][level] / max(proportions[model].values())
        if max(proportions[model].values()) > 0 else float('nan')
        for level in proportions[model]
    }
    for model in proportions
}

print("Proportion of Favorable Outcomes:")
for model, levels in proportions.items():
    for level, prop in levels.items():
        print(f"{model} - ChestPain {level}: {prop:.4f}")

print("\nDisparate Impact Ratio (DIR):")
for model, levels in dirs.items():
    for level, dir_ratio in levels.items():
        print(f"{model} - ChestPain {level} DIR: {dir_ratio:.4f}")

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.flatten()

for i, (level, group) in enumerate(results_cp.items()):
    tn_xgb, fp_xgb, fn_xgb, tp_xgb = calculate_confusion_matrix_components(group['True'], group['Predicted_XGB'])
    confusion_xgb = np.array([[tn_xgb, fp_xgb], [fn_xgb, tp_xgb]])
    sns.heatmap(confusion_xgb, annot=True, fmt='d', ax=axes[i * 2], cmap='Blues',
                xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    axes[i * 2].set_title(f'XGBoost Confusion Matrix - ChestPain {level}')
    
    tn_rf, fp_rf, fn_rf, tp_rf = calculate_confusion_matrix_components(group['True'], group['Predicted_RF'])
    confusion_rf = np.array([[tn_rf, fp_rf], [fn_rf, tp_rf]])
    sns.heatmap(confusion_rf, annot=True, fmt='d', ax=axes[i * 2 + 1], cmap='Blues',
                xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    axes[i * 2 + 1].set_title(f'Random Forest Confusion Matrix - ChestPain {level}')

plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report

for level, group in results_cp.items():
    print(f"\nXGBoost Classification Report - ChestPain {level}:")
    if len(group) > 0:
        print(classification_report(group['True'], group['Predicted_XGB'], zero_division=0))
    else:
        print("No data for this ChestPain level.")
    
    print(f"\nRandom Forest Classification Report - ChestPain {level}:")
    if len(group) > 0:
        print(classification_report(group['True'], group['Predicted_RF'], zero_division=0))
    else:
        print("No data for this ChestPain level.")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # disparate impact distilled models

# ## DIR distilled models for gender

# In[145]:


results = pd.DataFrame({
    'True': y_test,
    'Predicted_A': y_pred_A,
    'Predicted_B': y_pred_B
})

results['Gender'] = X_test['gender']  

results_male = results[results['Gender'] == 0]  
results_female = results[results['Gender'] == 1]  

def favorable_outcomes(tp, tn, total):
    return (tp + tn) / total

confusion_xgb_male = confusion_matrix(results_male['True'], results_male['Predicted_A'])
tn_xgb_m, fp_xgb_m, fn_xgb_m, tp_xgb_m = confusion_xgb_male.ravel()
prop_xgb_male = favorable_outcomes(tp_xgb_m, tn_xgb_m, len(results_male))

confusion_rf_male = confusion_matrix(results_male['True'], results_male['Predicted_B'])
tn_rf_m, fp_rf_m, fn_rf_m, tp_rf_m = confusion_rf_male.ravel()
prop_rf_male = favorable_outcomes(tp_rf_m, tn_rf_m, len(results_male))

confusion_xgb_female = confusion_matrix(results_female['True'], results_female['Predicted_A'])
tn_xgb_f, fp_xgb_f, fn_xgb_f, tp_xgb_f = confusion_xgb_female.ravel()
prop_xgb_female = favorable_outcomes(tp_xgb_f, tn_xgb_f, len(results_female))

confusion_rf_female = confusion_matrix(results_female['True'], results_female['Predicted_B'])
tn_rf_f, fp_rf_f, fn_rf_f, tp_rf_f = confusion_rf_female.ravel()
prop_rf_female = favorable_outcomes(tp_rf_f, tn_rf_f, len(results_female))

def disparate_impact_ratio(prop_male, prop_female):
    return prop_male / prop_female if prop_female != 0 else float('nan')

dir_xgb = disparate_impact_ratio(prop_xgb_male, prop_xgb_female)
dir_rf = disparate_impact_ratio(prop_rf_male, prop_rf_female)

print("Proportion of Favorable Outcomes:")
print(f"A - Male: {prop_xgb_male:.4f}, Female: {prop_xgb_female:.4f}")
print(f"B - Male: {prop_rf_male:.4f}, Female: {prop_rf_female:.4f}")
print("\nDisparate Impact Ratio (DIR):")
print(f"A DIR (Male/Female): {dir_xgb:.4f}")
print(f"B DIR (Male/Female): {dir_rf:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 12))

sns.heatmap(confusion_xgb_male, annot=True, fmt='d', ax=axes[0, 0], cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
axes[0, 0].set_title('XGBoost Confusion Matrix - Male')

sns.heatmap(confusion_xgb_female, annot=True, fmt='d', ax=axes[0, 1], cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
axes[0, 1].set_title('XGBoost Confusion Matrix - Female')

sns.heatmap(confusion_rf_male, annot=True, fmt='d', ax=axes[1, 0], cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
axes[1, 0].set_title('Random Forest Confusion Matrix - Male')

sns.heatmap(confusion_rf_female, annot=True, fmt='d', ax=axes[1, 1], cmap='Blues', 
            xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
axes[1, 1].set_title('Random Forest Confusion Matrix - Female')

plt.tight_layout()
plt.show()

print("\nA Classification Report - Male:")
print(classification_report(results_male['True'], results_male['Predicted_A']))

print("\nB Classification Report - Male:")
print(classification_report(results_male['True'], results_male['Predicted_B']))

print("\nA Classification Report - Female:")
print(classification_report(results_female['True'], results_female['Predicted_A']))

print("\nB Classification Report - Female:")
print(classification_report(results_female['True'], results_female['Predicted_B']))


# ## DIR for chestpain in distilled models

# In[146]:


results = pd.DataFrame({
    'True': y_test,
    'Predicted_A': y_pred_A,  
    'Predicted_B': y_pred_B   
})

results['ChestPain'] = X_test['chestpain']  

results_cp = {level: results[results['ChestPain'] == level] for level in [0, 1, 2, 3]}

def calculate_confusion_matrix_components(true_labels, predictions):
    confusion = confusion_matrix(true_labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = 0, 0, 0, 0
    if confusion.shape == (2, 2):
        tn, fp, fn, tp = confusion.ravel()
    elif confusion.shape == (1, 2):  
        tn, fp = confusion[0, :]
    elif confusion.shape == (2, 1):  
        fn, tp = confusion[:, 0]
    return tn, fp, fn, tp

def favorable_outcomes(tp, tn, total):
    return (tp + tn) / total if total > 0 else 0

proportions = {}
for model in ['Predicted_A', 'Predicted_B']:
    proportions[model] = {}
    for level, group in results_cp.items():
        tn, fp, fn, tp = calculate_confusion_matrix_components(group['True'], group[model])
        proportions[model][level] = favorable_outcomes(tp, tn, len(group))

dirs = {
    model: {
        level: proportions[model][level] / max(proportions[model].values())
        if max(proportions[model].values()) > 0 else float('nan')
        for level in proportions[model]
    }
    for model in proportions
}

print("Proportion of Favorable Outcomes:")
for model, levels in proportions.items():
    for level, prop in levels.items():
        print(f"{model} - ChestPain {level}: {prop:.4f}")

print("\nDisparate Impact Ratio (DIR):")
for model, levels in dirs.items():
    for level, dir_ratio in levels.items():
        print(f"{model} - ChestPain {level} DIR: {dir_ratio:.4f}")

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
axes = axes.flatten()

for i, (level, group) in enumerate(results_cp.items()):
    tn_a, fp_a, fn_a, tp_a = calculate_confusion_matrix_components(group['True'], group['Predicted_A'])
    confusion_a = np.array([[tn_a, fp_a], [fn_a, tp_a]])
    sns.heatmap(confusion_a, annot=True, fmt='d', ax=axes[i * 2], cmap='Blues',
                xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    axes[i * 2].set_title(f'Model A Confusion Matrix - ChestPain {level}')
    
    tn_b, fp_b, fn_b, tp_b = calculate_confusion_matrix_components(group['True'], group['Predicted_B'])
    confusion_b = np.array([[tn_b, fp_b], [fn_b, tp_b]])
    sns.heatmap(confusion_b, annot=True, fmt='d', ax=axes[i * 2 + 1], cmap='Blues',
                xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
    axes[i * 2 + 1].set_title(f'Model B Confusion Matrix - ChestPain {level}')

plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report

for level, group in results_cp.items():
    print(f"\nModel A Classification Report - ChestPain {level}:")
    if len(group) > 0:
        print(classification_report(group['True'], group['Predicted_A'], zero_division=0))
    else:
        print("No data for this ChestPain level.")
    
    print(f"\nModel B Classification Report - ChestPain {level}:")
    if len(group) > 0:
        print(classification_report(group['True'], group['Predicted_B'], zero_division=0))
    else:
        print("No data for this ChestPain level.")


# In[ ]:





# In[ ]:





# In[ ]:





# # visualized trees

# In[326]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def visualize_decision_tree(model_student, feature_names, class_names=None, export_path=None):
    plt.figure(figsize=(32, 6))
    
    plot_tree(
        model_student, 
        feature_names=feature_names,  
        class_names=class_names,      
        filled=True,                  
        rounded=True, 
        fontsize=10                   
    )

    if export_path:
        plt.savefig(export_path, bbox_inches='tight', format='png')  
        print(f"Graph saved to {export_path}")
    
    plt.title("Visualization of the Distilled Decision Tree")
    
    plt.show()


# In[353]:


visualize_decision_tree(modelA_soft, feature_names=feature_names, export_path='decision_treeA2.png')


# In[354]:


visualize_decision_tree(modelB_soft, feature_names=feature_names, export_path='decision_treeB2.png')

