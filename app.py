from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
# from sklearn.metrics import ,precision_score, recall_score
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

import pickle

def main():
    st.title("Epilepsy Detection Using EEG Signals")
    st.sidebar.title("Choose Classifier")
    # st.sidebar.markdown("Choose Classifier")
    # st.sidebar.subheader("Choose classifier")

if __name__ == '__main__':
    main()
    

file = st.file_uploader("Choose a file")
if file is None:
    file=='final_news.csv'
    
df=pd.read_csv('final_news.csv')


if st.sidebar.checkbox("Display data", False):
    st.subheader("EEG DATASET OF EPILEPTIC SEIZURE PATIENTS")
    st.write(df)

@st.cache_data(persist=True)
def split(df):
    x = df.iloc[:, 0:4].values
    y= df.iloc[:, -1].values 
    # x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, random_state=0)
    x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)  

    x_test = pickle.load(open('x_test.pkl', 'rb'))
    y_test = pickle.load(open('y_test.pkl', 'rb'))
    x_train = pickle.load(open('x_train.pkl', 'rb'))
    y_train = pickle.load(open('y_train.pkl', 'rb'))
    
    
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = split(df)

def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        plot_confusion_matrix(model, x_test, y_test, display_labels=   class_names)
        st.pyplot()
        
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        
        from sklearn.preprocessing import LabelBinarizer

        label_binarizer = LabelBinarizer().fit(y_train)
        y_onehot_test = label_binarizer.transform(y_test)
        
        plot_roc_curve(model, x_test, y_onehot_test)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        plot_precision_recall_curve(model, x_test, y_test)
        st.pyplot()
class_names = ["label 0", "label 1", "label 2"]

# st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine", "Logistic Regression","K-Nearest Neighbors","Decision Tree","Random Forest","CNN","GaussianNB","BernoulliNB","XGBoost"))


if classifier == "Support Vector Machine":
    # st.sidebar.subheader("Hyperparameters")
    # C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    # max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",""))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader(classifier +" Results")
        #model = LogisticRegression(random_state=0)
#         model.fit(x_train, y_train)
        model = pickle.load(open('SVC().pkl', 'rb'))
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Accuracy: ", accuracy.round(3))
        st.write("Precision: ", precision_score(y_test, y_pred,average='macro').round(3))
        st.write("Recall: ", recall_score(y_test, y_pred, average='macro').round(3))
        st.write("F1-Score: ", f1_score(y_test, y_pred, average='macro').round(3))
        plot_metrics(metrics)
st.set_option('deprecation.showPyplotGlobalUse', False)


if classifier == "Logistic Regression":
    # st.sidebar.subheader("Hyperparameters")
    # C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    # max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",""))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader(classifier +" Results")
        #model = LogisticRegression(random_state=0)
#         model.fit(x_train, y_train)
        model = pickle.load(open('LogisticRegression(random_state=0).pkl', 'rb'))
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Accuracy: ", accuracy.round(3))
        st.write("Precision: ", precision_score(y_test, y_pred,average='macro').round(3))
        st.write("Recall: ", recall_score(y_test, y_pred, average='macro').round(3))
        st.write("F1-Score: ", f1_score(y_test, y_pred, average='macro').round(3))
        plot_metrics(metrics)

if classifier == "K-Nearest Neighbors":
    # st.sidebar.subheader("Hyperparameters")
    # C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    # max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",""))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader(classifier +" Results")
        #model = LogisticRegression(random_state=0)
#         model.fit(x_train, y_train)
        model = pickle.load(open('KNeighborsClassifier().pkl', 'rb'))
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Accuracy: ", accuracy.round(3))
        st.write("Precision: ", precision_score(y_test, y_pred,average='macro').round(3))
        st.write("Recall: ", recall_score(y_test, y_pred, average='macro').round(3))
        st.write("F1-Score: ", f1_score(y_test, y_pred, average='macro').round(3))
        plot_metrics(metrics)



if classifier == "Random Forest":
    # st.sidebar.subheader("Hyperparameters")
    # C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    # max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",""))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader(classifier +" Results")
        #model = LogisticRegression(random_state=0)
#         model.fit(x_train, y_train)
        model = pickle.load(open('RandomForestClassifier().pkl', 'rb'))
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Accuracy: ", accuracy.round(3))
        st.write("Precision: ", precision_score(y_test, y_pred,average='macro').round(3))
        st.write("Recall: ", recall_score(y_test, y_pred, average='macro').round(3))
        st.write("F1-Score: ", f1_score(y_test, y_pred, average='macro').round(3))
        plot_metrics(metrics)
        

if classifier == "Decision Tree":
    # st.sidebar.subheader("Hyperparameters")
    # C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    # max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",""))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader(classifier +" Results")
        #model = LogisticRegression(random_state=0)
#         model.fit(x_train, y_train)
        model = pickle.load(open('DecisionTreeClassifier().pkl', 'rb'))
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Accuracy: ", accuracy.round(3))
        st.write("Precision: ", precision_score(y_test, y_pred,average='macro').round(3))
        st.write("Recall: ", recall_score(y_test, y_pred, average='macro').round(3))
        st.write("F1-Score: ", f1_score(y_test, y_pred, average='macro').round(3))
        plot_metrics(metrics)
        

if classifier == "GaussianNB":
    # st.sidebar.subheader("Hyperparameters")
    # C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    # max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",""))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader(classifier +" Results")
        #model = LogisticRegression(random_state=0)
#         model.fit(x_train, y_train)
        model = pickle.load(open('GaussianNB().pkl', 'rb'))
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Accuracy: ", accuracy.round(3))
        st.write("Precision: ", precision_score(y_test, y_pred,average='macro').round(3))
        st.write("Recall: ", recall_score(y_test, y_pred, average='macro').round(3))
        st.write("F1-Score: ", f1_score(y_test, y_pred, average='macro').round(3))
        plot_metrics(metrics)


if classifier == "BernoulliNB":
    # st.sidebar.subheader("Hyperparameters")
    # C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    # max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",""))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader(classifier +" Results")
        #model = LogisticRegression(random_state=0)
#         model.fit(x_train, y_train)
        model = pickle.load(open('BernoulliNB().pkl', 'rb'))
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Accuracy: ", accuracy.round(3))
        st.write("Precision: ", precision_score(y_test, y_pred,average='macro').round(3))
        st.write("Recall: ", recall_score(y_test, y_pred, average='macro').round(3))
        st.write("F1-Score: ", f1_score(y_test, y_pred, average='macro').round(3))
        plot_metrics(metrics)
        
if classifier == "XGBoost":
    # st.sidebar.subheader("Hyperparameters")
    # C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    # max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix",""))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader(classifier +" Results")
        #model = LogisticRegression(random_state=0)
#         model.fit(x_train, y_train)
        model = pickle.load(open('XGB.pkl', 'rb'))
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)

        st.write("Accuracy: ", accuracy.round(3))
        st.write("Precision: ", precision_score(y_test, y_pred,average='macro').round(3))
        st.write("Recall: ", recall_score(y_test, y_pred, average='macro').round(3))
        st.write("F1-Score: ", f1_score(y_test, y_pred, average='macro').round(3))
        plot_metrics(metrics)


# if classifier == "Decision Tree":
#     # st.sidebar.subheader("Hyperparameters")
#     # C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
#     # max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
#     metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
#     if st.sidebar.button("Classify", key="classify"):
#         st.subheader("K-Nearest Neighbors Results")
#         # model = LogisticRegression(random_state=0)
#         model = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
#         model.fit(x_train, y_train)
#         # model = pickle.load(open('DT.pkl', 'rb'))
#         accuracy = model.score(x_test, y_test)
#         y_pred = model.predict(x_test)

#         st.write("Accuracy: ", accuracy.round(3))
#         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(3))
#         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(3))
#         plot_metrics(metrics)
        
# if classifier == "Random Forest":
#     # st.sidebar.subheader("Hyperparameters")
#     # C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
#     # max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
#     metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
#     if st.sidebar.button("Classify", key="classify"):
#         st.subheader("Random Forest Results")
#         # model = LogisticRegression(random_state=0)
#         model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
#         model.fit(x_train, y_train)
#         # model = pickle.load(open('LR.pkl', 'rb'))
#         accuracy = model.score(x_test, y_test)
#         y_pred = model.predict(x_test)

#         st.write("Accuracy: ", accuracy.round(3))
#         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(3))
#         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(3))
#         plot_metrics(metrics)
        
# if classifier == "CNN":
#     # st.sidebar.subheader("Hyperparameters")
#     # C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
#     # max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
#     metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
#     if st.sidebar.button("Classify", key="classify"):
#         st.subheader("CNN Results")
#         # # model = LogisticRegression(random_state=0)
#         # model = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
#         # model.fit(x_train, y_train)
#         model = pickle.load(open('CNN.pkl', 'rb'))
#         # accuracy = model.score(x_test, y_test)
#         y_pred = model.predict(x_test)
#         for i in range(len(y_pred)):
#             y_pred[i]=y_pred[i].round()

#         st.write("Accuracy: ", accuracy_score(y_test, y_pred).round(3))
#         st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(3))
#         st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(3))
#         plot_metrics(metrics)
        
df1=pd.read_csv('report.csv',index_col=0)


if st.sidebar.checkbox("Display Report", False):
    st.subheader("Report")
    # st.snow()
    # st.balloons()
    st.write(df1)