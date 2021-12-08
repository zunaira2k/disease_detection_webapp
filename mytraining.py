import pandas as pd
import numpy as np
from sklearn.linear_model import  LogisticRegression
import pickle

def data_split(data, ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    test_set_size = int(len(data)* ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ =="__main__":
    df = pd.read_csv('finaldataset.csv')
    train, test = data_split(df, 0.3)
    x_train = train[['fever', 'tired', 'cough', 'diffBreath', 'sore_throat','bodyPain', 'runnyNose', 'diarreha', 'age']].to_numpy()
    x_test = test[['fever', 'tired', 'cough', 'diffBreath', 'sore_throat','bodyPain', 'runnyNose', 'diarreha', 'age']].to_numpy()

    y_train = train[['Infection_probability']].to_numpy().reshape(221760, )
    y_test = test[['Infection_probability']].to_numpy().reshape(95040, )

    clf = LogisticRegression(solver='lbfgs',class_weight='None', max_iter=221760)
    clf.fit(x_train, y_train)

    # open a file, where you ant to store the data
    file = open('model.pkl', 'wb')

    # dump information to that file
    pickle.dump(clf, file)
    file.close()

    

