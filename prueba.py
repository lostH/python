from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    irisdf_dummies.drop(["species"], axis=1), 
    irisdf_dummies.species, test_size=0.25, random_state=0
)

X_train.shape, X_test.shape, y_train.shape. y_test.shape

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)

knn.score(X_train,y_train)
knn.score(X_test,y_test)