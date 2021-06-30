#coding=utf-8
'''
https://stackoverflow.com/questions/54494785/sklearn-0-20-2-import-error-with-randomizedpca


@author: DMao
@time: 07.07.20    15:17
'''
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person= 60)
print(faces.target_names)
print(faces.images.shape)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
''''
fig, ax = plt.subplots(3, 5)
for i, axi in enumerate(ax.flat):
    axi.imshow(faces.images[i], cmap='bone')
    axi.set(xticks=[], yticks=[], xlabel = faces.target_names[faces.target[i]])
plt.show()
'''
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA as RandomizedPCA


# use PCA to extract features for SVM classifier
pca = RandomizedPCA(n_components=150, whiten=True, random_state=42)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)

from sklearn.model_selection import train_test_split
# X: data, y: target
Xtrain, Xtest,ytrain, ytest = train_test_split(faces.data, faces.target, random_state=42)

# grid search cross validation for tuning parameters
from sklearn.model_selection import GridSearchCV
param_grid = {'svc__C': [1,5,10,50], 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005]}
grid = GridSearchCV(model, param_grid)
model = grid.fit(Xtrain, ytrain)
# print(grid.best_params_) # only available after fit()
grid.best_params_
yfit = model.predict(Xtest)

'''
fig, ax = plt.subplots(4, 6)
for i, axi in enumerate(ax.flat):
    axi.imshow(Xtest[i].reshape(62, 47), cmap= 'bone')
    axi.set(xticks=[], yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],
                   color='black' if yfit[i] == ytest[i] else 'red')
fig.suptitle('Predicted Names; Incorrect Labels in Red', size=14)
'''



from sklearn.metrics import classification_report
repo = classification_report(ytest, yfit, target_names=faces.target_names)
print(repo)

from sklearn.metrics import confusion_matrix
mat = confusion_matrix(ytest, yfit)

sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=faces.target_names, yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()
