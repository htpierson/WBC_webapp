import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

st.set_option('deprecation.showPyplotGlobalUse', False)
def main():
    st.title("Binary Classification WebApp")    
    st.markdown("Do you have metastatic or benign cancer")

    st.sidebar.title("Binary Classification")
    st.sidebar.markdown("Do you have metastatic or benign cancer?")

    data = pd.read_csv('data.csv')
    y = data['diagnosis'].map({'B':0,'M':1})
    x = data.drop((['diagnosis','id', 'Unnamed: 32']), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)
    estimator = RandomForestClassifier(random_state = 42)
    n_features = st.sidebar.number_input("Select the amount of features you would like to use", 1, 30, key = 'n_features')
    selector = RFE(estimator, n_features_to_select = n_features, step = 1)
    selector = selector.fit(x_train, y_train)
    rfe_mask = selector.get_support()
    new_features = [] 
    for bool, feature in zip(rfe_mask, x_train.columns):
        if bool:
            new_features.append(feature)


    @st.cache(persist = True)
    def load_data():
        data = pd.read_csv('data.csv')
        data = data.drop((['Unnamed: 32']), axis=1)
        
        return data

    @st.cache(persist = True)
    def split(df):
        y = df['diagnosis'].map({'B':0,'M':1})
        x = df[new_features]
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

        return x_train, x_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels = class_names).plot(cmap = 'gist_heat_r')
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()

        if 'Precision-Recall Curve' in metrics_list:
            st.subheader("Precision-Recall Curve")
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()
            
    df = load_data()
    x_train, x_test, y_train, y_test = split(df)
    class_names = ['Benign', 'Metastatic']
    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression",
         "Random Forest",  "Multi-layer Perceptron (MLP)", "K Nearest Neighbors (KNN)", "Decision Tree", "AdaBoost"))
    
    if classifier == "Support Vector Machine (SVM)":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.1, 10.0, step = 0.1, key = 'C')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key = 'kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key = 'auto')
    
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C = C, kernel = kernel, gamma = gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)

    if classifier == "Logistic Regression":
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.1, 10.0, step = 0.1, key = 'C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key = 'max_iter')
        
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C = C, max_iter = max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)            

    if classifier == "Random Forest":
        st.sidebar.subheader("Model Hyperparameters")        
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step = 10, key = 'n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step = 1, key = 'max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key = 'bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
    
        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators = n_estimators, max_depth = max_depth, bootstrap = bootstrap, n_jobs = -1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)

    if classifier == "Multi-layer Perceptron (MLP)":
        st.sidebar.subheader("Model Hyperparameters")        
        alpha = st.sidebar.number_input("alpha (Regularization parameter)", 0.1, 10.0, step = 0.1, key = 'alpha')
        max_iter = st.sidebar.number_input("Maximum number of iterations", 50, 500, step = 10, key = 'max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Multi-layer Perceptron (MLP)")
            model = MLPClassifier(alpha = alpha, max_iter = max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics) 

    if classifier == "K Nearest Neighbors (KNN)":
        st.sidebar.subheader("Model Hyperparameters")        
        n_neighbors = st.sidebar.number_input("The number of neighbors", 1, 10, step = 1, key = 'n_neighbors')
        weights = st.sidebar.radio("Weights", ("uniform", "distance"), key = 'weights')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("K Nearest Neighbors (KNN)")
            model = KNeighborsClassifier(n_neighbors = n_neighbors, weights = weights)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)

    if classifier == "Decision Tree":
        st.sidebar.subheader("Model Hyperparameters")  
        criterion = st.sidebar.radio("Function to measure the quality of the split", ('gini', 'entropy'), 
        key = 'criterion') 
        max_depth = st.sidebar.number_input("Maximum depth of the tree", 1, 10, step = 1, key = 'max_depth')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("Decision Tree")
            model = DecisionTreeClassifier(criterion = criterion, max_depth = max_depth)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics)


    if classifier == "AdaBoost":
        st.sidebar.subheader("Model Hyperparameters")        
        n_estimators = st.sidebar.number_input("Number of Estimators", 10, 500, step = 10, key = 'n_estimators_ada')
        learning_rate = st.sidebar.number_input("Weight applied to each classifier", 0.1, 10.0, step = 0.1, key = 'learning_rate')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key = 'classify'):
            st.subheader("AdaBoost")
            model = AdaBoostClassifier(n_estimators = n_estimators, learning_rate = learning_rate)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels = class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels = class_names).round(2))
            plot_metrics(metrics) 

    
                        
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Wisconsin Breast Cancer Data Set (Classification)")
        st.write(df)
    
if __name__ == '__main__':
    main()