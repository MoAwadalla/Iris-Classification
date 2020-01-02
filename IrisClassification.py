#import libraries
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot

#assign data labels
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
#load in dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
dataset = read_csv(url, names = names)

#Dimensions of the dataset
print(dataset.shape)
#peak at data
print(dataset.head(20))
#data summary
print(dataset.describe())
#data distribution
print(dataset.groupby('class').size())

#Visualize data
#box and whisker plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()
#histograms
dataset.hist()
pyplot.show()
#scatter plot matrix
scatter_matrix(dataset)
pyplot.show()

#Split-out validation dataset
array = dataset.values
X = array[:,0:4]
y = array[:,4]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size = 0.2, random_state=1)

#Build the different models
models = []
#format: models.append((<label>, <algortithm>))
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
#evaluate each model
results = []
name = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    #print result
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

#based on result, SVM is the most accurate

#Make predictions using SVM
model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
#Evaluate predictions
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

