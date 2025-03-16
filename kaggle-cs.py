import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
file_path = '/home/users/ys468/ml kaggle/cs-671-fall-2024-final-project/train.csv'
df = pd.read_csv(file_path)
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    print(f"column_name: {col}, unique value: {df[col].nunique()}")

columns_to_drop = ['name', 'description', 'reviews', 'amenities','neighbourhood_cleansed','host_verifications','bathrooms_text']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

print(df.head())

df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
df['host_since_year_diff'] = 2024 - df['host_since'].dt.year
print(df[['host_since', 'host_since_year_diff']].head())


df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')


df['days_since_last_review'] = (pd.to_datetime('today') - df['last_review']).dt.days
df['days_since_first_review'] = (pd.to_datetime('today') - df['first_review']).dt.days


def categorize_review_period(days):
    if days <= 365:
        return 'within 1 year'
    elif days <= 1095:
        return '1-3 years'
    else:
        return 'more than 3 years'


df['review_last_category'] = df['days_since_last_review'].apply(categorize_review_period)
df['review_first_category'] = df['days_since_first_review'].apply(categorize_review_period)


print(df[['last_review', 'days_since_last_review', 'review_last_category']].head())
print(df[['first_review', 'days_since_first_review', 'review_first_category']].head())

room_type_counts = df['room_type'].value_counts()

print(room_type_counts)

neighbourhood_to_score = {
    'Manhattan': 5,
    'Brooklyn': 4,
    'Queens': 3,
    'Bronx': 2,
    'Staten Island': 1
}

df['neighbourhood_encoded'] = df['neighbourhood_group_cleansed'].map(neighbourhood_to_score)
df.head(5)

response_counts = df['host_response_time'].value_counts()
last_counts = df['review_last_category'].value_counts()
first_counts = df['review_first_category'].value_counts()

print(response_counts)
print(last_counts)
print(first_counts)

response_to_score = {
    'within an hour': 4,
    'within a few hours': 3,
    'within a day': 2,
    'a few days or more': 1,
}

df['response_encoded'] = df['host_response_time'].map(response_to_score)

last_to_score = {
    'within 1 year': 3,
    '1-3 years': 2,
    'more than 3 years': 1,
}
df['last_encoded'] = df['review_last_category'].map(last_to_score)

df_1 = pd.get_dummies(df, columns=['review_first_category', 'room_type'], 
                            prefix=['first_review', 'room_type'])



df_1.head(5)

df = df_1.drop(columns=['property_type', 'neighbourhood_group_cleansed','host_since','host_response_time','first_review','last_review','review_last_category'])

all_features = df.columns
print(all_features)

# check missing value
print("\nmissing value stats:")
print(df.isnull().sum())

# visualization

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# 找到缺失值比例大于 90% 的列并打印这些列名
columns_with_high_missing = missing_ratio[missing_ratio > 0.9].index

if len(columns_with_high_missing) > 0:
    print("Columns with missing value ratio > 90%:")
    print(columns_with_high_missing)
else:
    print("No columns have missing value ratio greater than 90%.")


columns_with_high_missing = missing_ratio[missing_ratio > 0.7].index

if len(columns_with_high_missing) > 0:
    print("Columns with missing value ratio > 70%:")
    print(columns_with_high_missing)
else:
    print("No columns have missing value ratio greater than 70%.")

df = df.drop(columns=[ 'has_availability','reviews_per_month','days_since_last_review','days_since_first_review', 'response_encoded'])

columns_to_fill = ['host_response_rate', 'host_acceptance_rate', 'review_scores_rating', 
                   'review_scores_accuracy', 'review_scores_cleanliness', 
                   'review_scores_checkin', 'review_scores_communication', 
                   'review_scores_location', 'review_scores_value']

for column in columns_to_fill:
    mode_value = df[column].mode()[0]  
    df[column].fillna(mode_value, inplace=True)  


missing_after_fill = df[columns_to_fill].isnull().sum()
print("Missing values after filling with mode:")
print(missing_after_fill)

df.head(5)

mode_value = df['host_is_superhost'].mode()[0]
df['host_is_superhost'].fillna(mode_value, inplace=True)

import matplotlib.pyplot as plt

# 假设你的 DataFrame 已经定义为 df
variables = ['accommodates', 'bathrooms', 'bedrooms', 'beds']

plt.figure(figsize=(15, 10))

for i, var in enumerate(variables, 1):
    plt.subplot(2, 2, i)
    df[var].hist(bins=20, edgecolor='black')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {var}')

plt.tight_layout()
plt.show()


columns_to_fill_mode = ['accommodates']
columns_to_fill_median = ['bathrooms', 'bedrooms', 'beds']

for column in columns_to_fill_mode:
    mode_value = df[column].mode()[0] 
    df[column].fillna(mode_value, inplace=True)


for column in columns_to_fill_median:
    median_value = df[column].median()  
    df[column].fillna(median_value, inplace=True)


missing_after_fill = df[columns_to_fill_mode + columns_to_fill_median].isnull().sum()

from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import matplotlib.pyplot as plt

# 假设目标变量为 'price'，其他列为特征
X = df.drop(columns=['price'])
y = df['price']

# 训练随机森林回归模型
model = RandomForestRegressor(n_estimators=100, random_state=100)
model.fit(X, y)

# 获取特征重要性并排序
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# 打印特征的重要性
print("Feature Importance:")
print(importance_df)

# 可视化特征重要性
plt.figure(figsize=(10, 8))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.gca().invert_yaxis()  # 反转Y轴以便最重要的特征在顶部
plt.show()

from pygam import LinearGAM, s, f
import pandas as pd
import matplotlib.pyplot as plt

# 假设目标变量为 'price'，其他列为特征
X = df.drop['price']
y = df['price']

# 只选择数值型特征来拟合 GAM 模型
numerical_features = X.select_dtypes(include=['number']).columns
X_numerical = X[numerical_features]

# 训练广义可加模型 (GAM)
gam = LinearGAM(s(0) + s(1) + s(2) + s(3) + s(4)).fit(X_numerical, y)

# 打印每个特征的统计信息
print("Summary of the GAM model:")
gam.summary()

# 可视化每个特征对目标变量的影响
for i, feature in enumerate(numerical_features):
    plt.figure()
    XX = gam.generate_X_grid(term=i)
    plt.plot(XX[:, i], gam.partial_dependence(term=i, X=XX), label=f'Partial Dependence of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Partial Effect on Price')
    plt.title(f'Partial Dependence Plot of {feature}')
    plt.legend()
    plt.show()
