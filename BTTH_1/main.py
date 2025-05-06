import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from scipy.stats import ttest_ind

df = pd.read_csv('E://Project//Python//BTTH_KHDL//BTTH_1//train.csv')

age_df = df[['Age', 'Pclass', 'Sex', 'SibSp', 'Parch']].copy()
age_df = pd.get_dummies(age_df, columns=['Sex'], drop_first=True)

known_age = age_df[age_df['Age'].notnull()]
unknown_age = age_df[age_df['Age'].isnull()]

linreg = LinearRegression()
linreg.fit(known_age.drop('Age', axis=1), known_age['Age'])

predicted_ages = linreg.predict(unknown_age.drop('Age', axis=1))
df.loc[df['Age'].isnull(), 'Age'] = predicted_ages

embarked_df = df[['Embarked', 'Pclass', 'Sex', 'Fare']].copy()
embarked_df = pd.get_dummies(embarked_df, columns=['Sex', 'Embarked'])

knn_imputer = KNNImputer(n_neighbors=3)
imputed = knn_imputer.fit_transform(embarked_df)

embarked_cols = [col for col in embarked_df.columns if col.startswith('Embarked')]
embarked_imputed = pd.DataFrame(imputed, columns=embarked_df.columns)
df['Embarked'] = embarked_imputed[embarked_cols].idxmax(axis=1).str.replace('Embarked_', '')

df['Deck'] = df['Cabin'].fillna('U').apply(lambda x: x[0])
df.drop('Cabin', axis=1, inplace=True)

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')
rare_titles = ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

df['FarePerPerson'] = df['Fare'] / df['FamilySize']

plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='Pclass', y='FarePerPerson', hue='Survived')
plt.title('Fare per person theo Pclass và Survived')
plt.show()

group1 = df[df['Survived'] == 0]['FarePerPerson']
group2 = df[df['Survived'] == 1]['FarePerPerson']
t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
print(f"T-test: t = {t_stat:.3f}, p = {p_value:.3f}")

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'Deck', 'FamilySize', 'IsAlone', 'Title', 'FarePerPerson']
X = df[features]
y = df['Survived']

numeric_features = ['Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'FarePerPerson']
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Deck', 'IsAlone', 'Title']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [5, 10, 15],
    'classifier__min_samples_split': [2, 5],
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print("Best params:", grid_search.best_params_)

best_model = grid_search.best_estimator_

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracy = cross_val_score(best_model, X, y, cv=cv, scoring='accuracy')
precision = cross_val_score(best_model, X, y, cv=cv, scoring='precision')
recall = cross_val_score(best_model, X, y, cv=cv, scoring='recall')
f1 = cross_val_score(best_model, X, y, cv=cv, scoring='f1')
roc_auc = cross_val_score(best_model, X, y, cv=cv, scoring='roc_auc')

print("Model Evaluation:")
print(f"Accuracy: {accuracy.mean():.3f}")
print(f"Precision: {precision.mean():.3f}")
print(f"Recall: {recall.mean():.3f}")
print(f"F1-score: {f1.mean():.3f}")
print(f"ROC AUC: {roc_auc.mean():.3f}")

importances = best_model.named_steps['classifier'].feature_importances_
feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_df = feature_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_df.head(15))
plt.title('Top 15 Feature Importance')
plt.tight_layout()
plt.show()
