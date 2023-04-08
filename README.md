# AUTOMATED-DIAGNOSIS-OF-EPILEPSY-USING-EEG-SIGNALS(powered by Intel oneAPI)

This project focuses on detecting an epilepsy disease,which is an  is a central nervous system (neurological) disorder in which brain activity becomes abnormal, causing seizures or periods of unusual behavior, sensations and sometimes loss of awareness,by providing DL & ML solution.


![Epilepsy](https://user-images.githubusercontent.com/90272634/230714090-4dd97c02-fdba-4b39-900b-2ba8f43199f2.jpg) <pre> ![Epilepsy_2](https://user-images.githubusercontent.com/90272634/230714210-9f8019ca-27fb-417d-b196-d6ccb8501470.jpg)</pre>

**Problem statement:** Epilepsy is a neurological disorder characterized by recurrent seizures. Despite advances in diagnostic techniques and treatments, epilepsy remains a challenging disease to diagnose accurately and manage effectively. There is a need for a reliable and non-invasive method of detecting epilepsy that can aid in early diagnosis and treatment, and improve the quality of life for those living with this condition. The development of an accurate and effective method for detecting epilepsy could potentially reduce healthcare costs and improve patient outcomes.

**Solution:** The solution for the above problem is developing a 1D CNN model capable of detecting epilepsy, training the model with a training and validation dataset, evaluating the model's performance with a separate testing dataset, and tuning the hyperparameters to optimize the model's performance. Using a 1D CNN for detecting epilepsy has shown promising results in accurately detecting seizures by capturing temporal features in the data.And checking the detection with other models to find the best model for classification between normal person and ectal person.

# WHY ONEAPI
![logo-oneapi-rwd](https://user-images.githubusercontent.com/90272634/230717338-f2dc33e7-31df-4dc0-98f1-9074cb7252e2.png)

Intel oneAPI is a suite of development tools for creating software that can run on a variety of platforms, including CPUs, GPUs, and FPGAs. It includes a number of components, such as compilers, libraries, and tools for performance analysis and debugging. The advantage of using oneAPI is that it allows developers to write code once and run it on a variety of hardware platforms, which can save time and effort.

## Toolkit used: Intel® AI Analytics Toolkit (AI Kit)

In this project we have used different machine learning models which **increses the the runtime when we take time series data of EEG signal from brain through sensors( small metal discs also called EEG electrodes ) after filtering and sampling the signals**.This causes delay in processing the data in the systems with limited processing power.**The Intel® AI Analytics Toolkit (AI Kit) helps to resolve this problem by providing better results by optimising the models**.We use deep learning API like keras which  are optimized for the Intel architecture by the oneAPI platform and further boosts the inference of the models.Scikit-learn (sklearn) is a popular Python library for machine learning, but it may not be optimized for Intel hardware by default.By integrating oneAPI into scikit-learn, the performance of some of its algorithms, such as linear regression, can be significantly improved.**integrating oneAPI into scikit-learn can improve its performance on Intel hardware by taking advantage of highly optimized libraries and routines for mathematical computations and data analysis**.

![image](https://user-images.githubusercontent.com/90272634/230718814-9bd28e3b-3ae1-4bf0-9641-39a7c7fc4dd5.png)

![image](https://user-images.githubusercontent.com/90272634/230719024-aa1635f7-2f7e-4399-a209-fcd05cf43b06.png)
Epilepsy detection model is executed on Intel DevCloud.And Intel Extension for Scikit-Learn is enabled.
Intel DevCloud Link:https://jupyter.oneapi.devcloud.intel.com/user/u190596/lab/workspaces/auto-e/tree/1DCNN.ipynb




