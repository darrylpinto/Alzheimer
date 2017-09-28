
import subprocess

import pandas as pd
from sklearn import tree, model_selection, metrics, neighbors


def visualize_tree(decision_tree, feature_names, dot_loc="",
                   decision_tree_viz_path=""):
    """
    Method to visualize the tree

    This method requires dot.exe

    dot is the command used in linux to visualize a graph
    The decision tree is saved in .dot file
    and using the dot command the tree is drawn

    :param decision_tree: The decision Tree
    :param feature_names: The list of features
    :param dot_loc: The location of dot.exe
    :param decision_tree_viz_path: Path where the decision
                                tree needs to be stored
            if decision_tree_viz_path = "", it will be saved in the same directory as the code

    :return: None
    """

    with open(decision_tree_viz_path + "DecisionTree.dot", 'w') as file:
        tree.export_graphviz(decision_tree, out_file=file,
                             feature_names=feature_names)

    try:

        # Command to convert .dot file to required format
        command = [dot_loc, "-Tpdf",
                   decision_tree_viz_path + "DecisionTree.dot", "-o",
                   decision_tree_viz_path + "DecisionTree.pdf"]
        try:
            subprocess.check_call(command)
        except subprocess.CalledProcessError:
            print("Cannot perform Visualization")

    except FileNotFoundError:
        print("dot.exe is not present! Cannot perform Visualization")


def encode_classifier(df, classifier):
    """
    Method to encode the classifier
    This method converts categorical data to numerical data

    :param df: Data frame
    :param classifier: The column used as classifier
    :return: None
    """

    encoder = {"Early": 0, "Middle": 1, "Late": 2}
    df["Classifier"] = df[classifier].replace(encoder)


def decision_tree_classifier(df, features):
    """
    Method to create the decision tree classifier using features
    :param df: Data frame
    :param features: List of features of the data
    :return: None
    """

    # The location of dot.exe on the user system
    dot_loc = "C:\\Program Files\\graphviz-2.38\\release\\bin\\dot.exe"
    # dot_loc = input("Enter the location of dot.exe"
    #                 " \n(For example:       C:\\Program Files\\"
    #                 "graphviz-2.38\\release\\bin\\dot.exe):")
    # dot_loc = dot_loc.strip()

    # Creates a new Decision tree object
    decision_tree = tree.DecisionTreeClassifier(criterion="entropy",
                                                max_depth=4,
                                                # )
                                                min_samples_split=6)

    y = df["Classifier"]
    x = df[features]

    print("\nDecision Tree")

    # Prepares Stratified Training and Testing Data
    x_train, x_test, y_train, y_test = model_selection. \
        train_test_split(x, y, train_size=0.6)

    # Training of Decision Tree
    decision_tree.fit(x_train, y_train)

    # Visualization of Decision Tree
    visualize_tree(decision_tree, features, dot_loc=dot_loc)

    # Accuracy of decision Tree Model
    accuracy = decision_tree.score(x_test, y_test)
    print("Accuracy using training  split: %0.5f" % (accuracy * 100))

    # Confusion Matrix
    confusion = metrics.confusion_matrix(y_test, decision_tree.predict(x_test),
                                         labels=[0, 1, 2])
    print("Confusion matrix:", confusion, sep="\n")

    # Cross-Validation for Consistent Accuracy
    scores = model_selection.cross_val_score(decision_tree, X=x, y=y, cv=10)
    print("Accuracy using cross validation: %0.5f" % (scores.mean() * 100))


def knn(df, features):
    knn = neighbors.KNeighborsClassifier(n_neighbors=11)
    y = df["Classifier"]
    x = df[features]

    # Prepares Stratified Training and Testing Data
    x_train, x_test, y_train, y_test = model_selection. \
        train_test_split(x, y, train_size=0.6)
    print("\nKNN")
    knn.fit(X=x_train, y=y_train)
    accuracy = knn.score(x_test, y_test)
    print("Accuracy using training-testing split: %0.5f" % (accuracy * 100))

    confusion = metrics.confusion_matrix(y_test, knn.predict(x_test),
                                         labels=[0, 1, 2])
    print("Confusion matrix:", confusion, sep="\n")

    # Cross-Validation for Consistent Accuracy
    scores = model_selection.cross_val_score(knn, X=x, y=y, cv=10)
    print("Accuracy using cross validation: %0.5f" % (scores.mean() * 100))




def main():
    """
    The main method
    :return: None
    """

    # Path where the images are saved
    path = "D:\\Studies\\Sem 2\\Intelligent Systems\\Project\\Data\\"
    sep = ''

    # Name of the file where the features are stored
    filename = "Final_file.csv"

    df = pd.read_csv(path + sep + filename, index_col=0)
    df.columns = df.columns.str.strip()

    df["Alzheimer_Stage"] = df["Alzheimer_Stage"].str.strip()

    encode_classifier(df, "Alzheimer_Stage")
    features = list(df.columns[0:3])

    decision_tree_classifier(df, features)

    knn(df, features)


if __name__ == "__main__":
    main()
