import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

gender_submition = pd.read_csv(r'gender_submission.csv')
data_test = pd.read_csv(r'test.csv')
data_train = pd.read_csv(r'train.csv')
pd.set_option('display.max_columns', None)
#print(data_train.head())
#print(data_train.shape)
#print(data_test.shape)
#print(gender_submition.shape)
#print(data_train.isnull().sum())

def edit_Sex(data):
    #замена female на 0 и male на 1
    data.loc[data['Sex'] == 'female', 'Sex'] = 0
    data.loc[data['Sex'] == 'male', 'Sex'] = 1

def edit_Embarked(data):
    # замена S на 0 и C на 1 и Q на 2
    data.loc[data['Embarked'] == 'S', 'Embarked'] = 0
    data.loc[data['Embarked'] == 'C', 'Embarked'] = 1
    data.loc[data['Embarked'] == 'Q', 'Embarked'] = 2
    data.Embarked = data.Embarked.fillna(0)

def edit_name(data):
    data['Title'] = data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    data.Title = data.Title.replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major',
                                     'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    data.Title = data.Title.replace('Mlle', 'Miss')
    data.Title = data.Title.replace('Ms', 'Miss')
    data.Title = data.Title.replace('Mme', 'Mrs')

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    data.Title = data.Title.map(title_mapping)
    data.Title = data.Title.fillna(0)

    return data

'''
plt.subplot(121)
plt.hist(data_train.Age, 50)
plt.show()
'''

def edit_age_random_array(data):
    data = data.dropna()
    data = data.to_numpy()
    hist, bins = np.histogram(data, bins=50)

    bin_midpoints = bins[:-1] + np.diff(bins) / 2
    cdf = np.cumsum(hist)
    cdf = cdf / cdf[-1]
    values = np.random.rand(177)
    value_bins = np.searchsorted(cdf, values)
    random_from_cdf = bin_midpoints[value_bins]

    plt.subplot(121)
    plt.hist(data, 50)
    plt.subplot(122)
    plt.hist(random_from_cdf, 50)
    plt.show()

    return random_from_cdf

data_Age = data_train.Age
#edit_age_random_array(data_Age)
'''
print(pd.qcut(data_train['Age'], q=6))
print(pd.qcut(data_train['Fare'], q=4))
'''
def fill_age(data):
    data.Age = data.Age.interpolate()
    data.Age = np.around(data.Age, 1)

    data.loc[data['Age'] <= 18.0, 'Age'] = 0
    data.loc[(data['Age'] > 18.0) & (data['Age'] <= 24), 'Age'] = 1
    data.loc[(data['Age'] > 24) & (data['Age'] <= 28), 'Age'] = 2
    data.loc[(data['Age'] > 28) & (data['Age'] <= 34), 'Age'] = 3
    data.loc[(data['Age'] > 34) & (data['Age'] <= 43), 'Age'] = 4
    data.loc[(data['Age'] > 43) & (data['Age'] <= 90), 'Age'] = 5

def edit_Fare(data):
    data.loc[data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31.0), 'Fare'] = 2
    data.loc[(data['Fare'] > 31.0) & (data['Fare'] <= 520.0), 'Fare'] = 3

edit_Fare(data_train)
edit_Fare(data_test)

edit_Sex(data_test)
edit_Sex(data_train)

edit_name(data_test)
edit_name(data_train)

data_train = data_train.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
data_test = data_test.drop(['Name', 'Ticket', 'Cabin',], axis=1)

edit_Embarked(data_train)
edit_Embarked(data_test)

fill_age(data_train)
fill_age(data_test)
data_test.Fare = data_test.Fare.interpolate()

all_data = pd.concat([data_train, data_test])


'''
plt.subplot(121)
plt.hist(data_train.Age, 50)
plt.show()
'''

'''

sns.barplot(x = data_train.Sex, y = data_train.Survived, color='black')
plt.show()

sns.barplot(x = data_train.Title, y = data_train.Survived, color='orange')
plt.show()

sns.barplot(x = data_train.Pclass, y = data_train.Survived, color='purple')
plt.show()

data_train['Age_cat'] = pd.qcut(data_train.Age,7)
sns.catplot(data = data_train,hue = 'Survived', x = 'Age_cat', kind='count', saturation=0.5)
plt.xticks(rotation=45)
plt.show()
'''

class model():
    def __init__(self, train_data, test_data):
        self.train_data_X = train_data.drop(['Survived'], axis=1)
        self.train_data_y = train_data['Survived']
        self.test_data = test_data.drop('PassengerId', axis=1)


    def DecisionTree_Model(self):
        parametrs = {'criterion':['entropy'], 'max_depth':[1, 20], 'min_samples_split':[2, 20], 'min_samples_leaf':[1, 10]}
        clf_f = tree.DecisionTreeClassifier()

        grid_search_clf_f = GridSearchCV(clf_f, parametrs, cv=5, n_jobs=-1)
        grid_search_clf_f.fit(self.train_data_X, self.train_data_y)

        pred_test_y = grid_search_clf_f.predict(self.test_data)

        #print(type(pred))
        #print(grid_search_clf_f.best_estimator_)
        print('DecisionTree Score:', round(grid_search_clf_f.best_score_ * 100, 2), 'percent')
        return pred_test_y

    def RandomForest_model(self):

        parametrs = {'n_estimators': range(10, 51, 10),
                     'max_depth': range(1, 10, 2),
                     'min_samples_leaf': range(1, 10, 1),
                     'min_samples_split': range(2, 10, 2)}

        clf_rf = RandomForestClassifier(random_state=1)
        grid_search_cv_clf_rf = GridSearchCV(clf_rf, parametrs, cv=5, n_jobs=-1)
        grid_search_cv_clf_rf.fit(self.train_data_X, self.train_data_y)

        pred_test_y = grid_search_cv_clf_rf.predict(self.test_data)

        pred_train_y = grid_search_cv_clf_rf.predict(self.train_data_X)

        conf_matrix = confusion_matrix(self.train_data_y, pred_train_y)
        ax = sns.heatmap(conf_matrix, annot=True, cmap='Reds')

        ax.set_title('Seaborn Confusion Matrix for RandomForestClassifier_Model')
        ax.set_xlabel('\nPredicted Survival')
        ax.set_ylabel('Actual Survival ')

        ax.xaxis.set_ticklabels(['False', 'True'])
        ax.yaxis.set_ticklabels(['False', 'True'])
        plt.show()

        #predict = grid_search_cv_clf_rf.predict(self.test_data)
        #print('Random Forest Score:', round(grid_search_cv_clf_rf.best_score_ * 100, 2), 'percent')
        return pred_test_y

    def LinearRegression_model(self):

        parameters = {'fit_intercept':[True,False], 'copy_X':[True, False]}
        LinRg = LinearRegression()

        grid_search_cv_Lin_rg = GridSearchCV(LinRg, parameters, cv=5, n_jobs=-1)
        grid_search_cv_Lin_rg.fit(self.train_data_X, self.train_data_y)

        print('LinearRegression Score:', round(grid_search_cv_Lin_rg.best_score_ * 100, 2), 'percent')


    def LogisticRegression_model(self):

        parameters = [{'C': [0.001, 0.01, 0.1, 1, 10, 100]}]
        LogisticRg = LogisticRegression()

        grid_search_cv_LogisticRg = GridSearchCV(LogisticRg, parameters, cv=5, n_jobs=-1)
        grid_search_cv_LogisticRg.fit(self.train_data_X, self.train_data_y)

        print('LogisticRegression Score:', round(grid_search_cv_LogisticRg.best_score_ * 100, 2), 'percent')


    def SVC_model(self):
        X_train, X_test, y_train, y_test = train_test_split(self.train_data_X, self.train_data_y, test_size=0.2)

        SVCrg = SVC()
        SVCrg.fit(X_train, y_train)

        print('SVC Score:', round(SVCrg.score(X=X_test, y=y_test) * 100, 2) , 'percent')


    def KNeighborsClassifier_model(self):
        parametrs = {'n_neighbors': [3, 5, 7, 9, 11]}

        KNeighbors = KNeighborsClassifier()

        grid_search_KNeighbors = GridSearchCV(KNeighbors, parametrs, cv=5, n_jobs=-1)
        grid_search_KNeighbors.fit(self.train_data_X, self.train_data_y)

        print('KNeighbors Score:', round(grid_search_KNeighbors.best_score_ * 100, 2), 'percent')

    def GaussianNB_model(self):
        parametrs = {'var_smoothing': np.logspace(0, -9, num=100)}

        GaussianNBrf = GaussianNB()

        grid_search_GaussianNBrf = GridSearchCV(GaussianNBrf, parametrs, cv=5, n_jobs=-1)
        grid_search_GaussianNBrf.fit(self.train_data_X, self.train_data_y)

        print('GaussianNB Score:', round(grid_search_GaussianNBrf.best_score_ * 100, 2), 'percent')


model = model(train_data=data_train, test_data=data_test)
pred_test = model.RandomForest_model()

data = {'PassengerId': data_test['PassengerId'], 'Survived': pred_test}
df_output = pd.DataFrame(data, columns=['PassengerId', 'Survived'])
df_output.to_csv('data.csv', index=False)
#print(df_output)

