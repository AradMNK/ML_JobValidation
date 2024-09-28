import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler

def load_data_train(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    
    income_95th_percentile = df['Income'].quantile(0.98)
    df = df[df['Income'] <= income_95th_percentile]
    age_95th_percentile = df['Age'].quantile(0.98)
    df = df[df['Age'] <= age_95th_percentile]
    
    df_processed = pd.DataFrame(df)

    df_processed['MF'] = df['MF'].map({'M': 1, 'F': 0})

    education_mapping = {'Ad. Dip': 1, 'Dip': 2, 'Bach': 3, 'Mst': 4, 'Doct': 5, 'P. Doct': 6}
    df_processed['LoE'] = df['LoE'].map(education_mapping)

    housing_mapping = {'N': 0, 'R': 1, 'O': 2}
    df_processed['Housing'] = df['Housing'].map(housing_mapping)

    df_processed['Car'] = df['Car'].map({True: 1, False: 0})
    
    X = df.drop(columns=['Res', 'Id'])
    y = df['Res']
    return X, y

def load_data_test(file_path):
    df = pd.read_csv(file_path)
    df.dropna(inplace=True)
    
    df_processed = pd.DataFrame(df)

    df_processed['MF'] = df['MF'].map({'M': 1, 'F': 0})

    education_mapping = {'Ad. Dip': 1, 'Dip': 2, 'Bach': 3, 'Mst': 4, 'Doct': 5, 'P. Doct': 6}
    df_processed['LoE'] = df['LoE'].map(education_mapping)

    housing_mapping = {'N': 0, 'R': 1, 'O': 2}
    df_processed['Housing'] = df['Housing'].map(housing_mapping)

    df_processed['Car'] = df['Car'].map({True: 1, False: 0})
    
    X = df.drop(columns=['Res', 'Id'])
    y = df['Res']
    return X, y

def scale(X):
    scaler = MinMaxScaler()
    
    numerical_columns = ['Age', 'YoW', 'YoCW', 'Income']
    X[numerical_columns] = scaler.fit_transform(X[numerical_columns])

    return X, scaler

def train_random_forest(X_train, y_train):
    
    best_params = {'max_depth': None,
    'max_features': 'sqrt',
    'min_samples_leaf': 1,
    'n_estimators': 1000}
    
    clf = RandomForestClassifier(**best_params)
    clf.fit(X_train, y_train)
    return clf

def main():
    #### This is where you can change if the reading does not work
    X_train, y_train = load_data_train('train_data.csv')
    
    X_train_processed, scaler = scale(X_train)
    y_train = y_train.map({'Accept': 1, 'Reject': 0})

    clf = train_random_forest(X_train_processed, y_train)

    #### This is where you can change if the reading does not work
    X_test, y_test = load_data_test('test_data_2.csv')
    
    
    y_test = y_test.map({'Accept': 1, 'Reject': 0})
    
    numerical_columns = ['Age', 'YoW', 'YoCW', 'Income']
    X_test_scaled = pd.DataFrame(X_test)
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

    y_pred_proba = clf.predict_proba(X_test_scaled)[:, 1]

    auprc = average_precision_score(y_test, y_pred_proba)
    print(auprc)

if __name__ == "__main__":
    main()