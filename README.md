# AUTOMATED-DIAGNOSIS-OF-EPILEPSY-USING-EEG-SIGNALS(powered by Intel oneAPI)

This project focuses on detecting an epilepsy disease,which is an  is a central nervous system (neurological) disorder in which brain activity becomes abnormal, causing seizures or periods of unusual behavior, sensations and sometimes loss of awareness,by providing DL & ML solution.

![Epilepsy_2](https://user-images.githubusercontent.com/90272634/230714210-9f8019ca-27fb-417d-b196-d6ccb8501470.jpg) ![Epilepsy](https://user-images.githubusercontent.com/90272634/230714090-4dd97c02-fdba-4b39-900b-2ba8f43199f2.jpg)

**Problem statement:** Epilepsy is a neurological disorder characterized by recurrent seizures. Despite advances in diagnostic techniques and treatments, epilepsy remains a challenging disease to diagnose accurately and manage effectively. There is a need for a reliable and non-invasive method of detecting epilepsy that can aid in early diagnosis and treatment, and improve the quality of life for those living with this condition. The development of an accurate and effective method for detecting epilepsy could potentially reduce healthcare costs and improve patient outcomes.

**Solution:** The solution for the above problem is developing a 1D CNN model capable of detecting epilepsy, training the model with a training and validation dataset, evaluating the model's performance with a separate testing dataset, and tuning the hyperparameters to optimize the model's performance. Using a 1D CNN for detecting epilepsy has shown promising results in accurately detecting seizures by capturing temporal features in the data.And checking the detection with other models to find the best model for classification between normal person and ectal person.

# WHY ONEAPI
![logo-oneapi-rwd](https://user-images.githubusercontent.com/90272634/230717338-f2dc33e7-31df-4dc0-98f1-9074cb7252e2.png)

Intel oneAPI is a suite of development tools for creating software that can run on a variety of platforms, including CPUs, GPUs, and FPGAs. It includes a number of components, such as compilers, libraries, and tools for performance analysis and debugging. The advantage of using oneAPI is that it allows developers to write code once and run it on a variety of hardware platforms, which can save time and effort.

## Toolkit used: Intel® AI Analytics Toolkit (AI Kit)

In this project we have used different machine learning models which **increses the the runtime when we take time series data of EEG signal from brain through sensors( small metal discs also called EEG electrodes ) after filtering and sampling the signals**.This causes delay in processing the data in the systems with limited processing power.**The Intel® AI Analytics Toolkit (AI Kit) helps to resolve this problem by providing better results by optimising the models**.We use deep learning API like keras which  are optimized for the Intel architecture by the oneAPI platform and further boosts the inference of the models.Scikit-learn (sklearn) is a popular Python library for machine learning, but it may not be optimized for Intel hardware by default.By integrating oneAPI into scikit-learn, the performance of some of its algorithms, such as linear regression, can be significantly improved.**integrating oneAPI into scikit-learn can improve its performance on Intel hardware by taking advantage of highly optimized libraries and routines for mathematical computations and data analysis**.

![image](https://user-images.githubusercontent.com/90272634/230718814-9bd28e3b-3ae1-4bf0-9641-39a7c7fc4dd5.png)

![image](https://user-images.githubusercontent.com/90272634/230719024-aa1635f7-2f7e-4399-a209-fcd05cf43b06.png)

![image](https://user-images.githubusercontent.com/90272634/230719485-320a776f-ef17-49a0-abc4-b27ffb9d1937.png)

![image](https://user-images.githubusercontent.com/90272634/230719213-00ca137d-9769-4909-bc90-be4df1498f73.png)

Epilepsy detection model is executed on Intel DevCloud where the model is optimised by oneDNN .And Intel Extension for Scikit-Learn is enabled.The AI Analytics Toolkit is used to install and optimize all the libraries which are present in the project.

Intel DevCloud Link:https://jupyter.oneapi.devcloud.intel.com/user/u190596/lab/workspaces/auto-e/tree/1DCNN.ipynb

# Time Elapsed

![image](https://user-images.githubusercontent.com/90272634/230719941-3633f251-a2b7-4174-9ce7-7d96af44a945.png)

Time taken for the project to execute all the machine learning models without oneAPI in google colab :0.8689255714416504 sec

![image](https://user-images.githubusercontent.com/90272634/230720064-be221a2f-fe41-439e-b5c4-44464bf3e720.png)

Time taken for the project to execute all the machine learning models with oneAPI in oneapi environment:0.15295076370239258 sec

Hence, we observe a difference of 0.715974808 seconds which is obtained with the help of oneAPI libraries.

# Results and discussion

After Traning and testing of different models we got more accuracy for the 1D CNN model with accuracy of 99.00% for given data.Whereas the  SVM ,GaussianNB, LogisticRegression, BernoulliNB,KNeighborsClassifier,DecisionTreeClassifier,RandomForestClassifier and XGBRFClassifierperformed with an acuracy of 80.8%,71.2%,81.6%,76.0%,82.39%,80.8%,83.2% and 88.8% The oneAPI reduced the overall runtime and GPU usage significantly compared to normal platforms.
All the models haven been optimised and execution time have been reduced by using Intel oneAPI.

### ✅Graphs

![image](https://user-images.githubusercontent.com/90272634/230721127-d1a88f5d-c564-4152-8494-c6dbbf762407.png)

Accuracy of the model is increasing in both testing and training as number of epoch increases.

![image](https://user-images.githubusercontent.com/90272634/230721219-350cf204-1c03-4925-887e-27334327c11a.png)

model loss is decreasing in both testing and training as number of epoch increases.

![image](https://user-images.githubusercontent.com/90272634/230721300-41e47a5f-ef63-40e4-93a6-cbff3a90232b.png)

Confusion matrix , f1 score of 1D CNN optimised by the Intel oneAPI.

![image](https://user-images.githubusercontent.com/90272634/230721486-73195fbb-2c67-45e0-86a1-adc7f9f48fe7.png)

![image](https://user-images.githubusercontent.com/90272634/230721718-a13a8de8-465f-4b02-ae04-ba9253838453.png)

Accuracy and Confusion matrix of different models optimised by oneAPI tool box.

# HOW TO RUN THIS PROJECT ![image](https://user-images.githubusercontent.com/90272634/230730418-d000dbc9-febc-4be2-9ad3-ad2821f56404.png)

### ✅STEP 1: Download the models,datasheets and matlab codes from the resource.
### ✅STEP 2: Clone the GitHub repositort.
### ✅STEP 2: Import datasheet into matlab and run the code for extracting features,Which are used in machine learning model training and testing.
### ✅STEP 3: Store all the extracted features into a simgle .csv file.
### ✅STEP 4: Create a new Conda environment and install the required libraries through the Intel channel.
### ✅STEP 5: Execute the code in Intel oneAPI cloud platform so that execution time and GPU will be usage significantly reduced.

# PROJECT VISUALIZATION USING WEBAPP
![Screenshot (377)](https://user-images.githubusercontent.com/110721429/230741086-da22d9ca-3a10-4f57-8c2f-3abdf99fa1bf.png)

(Demo: https://drive.google.com/file/d/1cxti4d01uxlF6fqGymEDRFfOt6xrvVra/view?usp=share_link)

For the given dataset the machine learning algorithms are applied and evaluation metrics such as confusion matrix, accuracy, precision, recall, f1-score are displayed in a web application using StreamLit package.

# What I learned ![image](https://user-images.githubusercontent.com/72274851/218499685-e8d445fc-e35e-4ab5-abc1-c32462592603.png)

![image](https://user-images.githubusercontent.com/72274851/220130227-3c48e87b-3e68-4f1c-b0e4-8e3ad9a4805a.png)

✅Building application using intel oneAPI:Intel oneAPI is a suite of development tools for creating software that can run on a variety of platforms..Scikit-learn (sklearn) is a popular Python library for machine learning, but it may not be optimized for Intel hardware by default.By integrating oneAPI into scikit-learn deployment of code into hardware will become more compatable.

✅Building a  Epilepsy detection application involves a significant amount of research and development. During the process, I likely learned a number of things, including: 

✅Machine Learning: I likely learned about different machine learning algorithms and how they can be applied to predict epilepsy by extracting the features.

✅Data Analysis: I likely gained experience in collecting and analyzing large amounts of data, including EEG signal data, to train our machine learning models.

✅Collaboration: Building a project like this likely required collaboration with a team of experts in various fields, such as VLSI, machine learning,deep learning and data analysis, and I likely learned the importance of working together to achieve common goals.

These are just a few examples of the knowledge and skills that i likely gained while building this project. 

Overall, building a epilepsy detecting device is a challenging and rewarding experience that requires both hardwarw and software knowledge on VLSI,IoT,machine learning and deep learning.This is my step to complete this challenging project by building a model for detection of epilepsy and we are in plan of deploying this codel in fpga with help of oneAPI and make an SoC(System on Chip) to make this prototype as wearable device.





















