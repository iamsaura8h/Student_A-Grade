
# Create your views here.
from ast import alias
from concurrent.futures import process
from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages

import Student_Performance

from .forms import UserRegistrationForm
from .models import UserRegistrationModel
from django.conf import settings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import datetime as dt
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier


# Create your views here.


def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})

def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not Yet Activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):

    return render(request, 'users/UserHomePage.html', {})
#===========================================================================



def DatasetView(request):
    path = settings.MEDIA_ROOT + "//" + 'student.csv'
    df = pd.read_csv(path, nrows=1000)
    df = df.to_html
    return render(request, 'users/viewdataset.html', {'data': df})

def ml(request):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    import sklearn.metrics as metrics
    import numpy as np 
    from sklearn.metrics import accuracy_score, f1_score, r2_score, confusion_matrix, classification_report
    from sklearn.svm import SVC  
    from sklearn.neighbors import KNeighborsClassifier 

    data = pd.read_csv(r'D:\Personal Projects\StudentPerformance\media\student.csv')
    print(data.head())

    data['grade'] = data['grade'].replace(['>90','<90'],[1,0])

    from sklearn.preprocessing import LabelEncoder                
    
    lb = LabelEncoder()
    data['gender']= lb.fit_transform(data['gender'])
    data['course_level'] = lb.fit_transform(data['course_level'])

    y = data['grade']
    X = data.drop('grade', axis=1)

    # Split the data into training and validation sets
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=7)

    model = RandomForestClassifier(random_state=7, n_estimators=100)
    model.fit(train_X, train_y)

    # Predict classes given the validation features
    pred_y = model.predict(val_X)

    # Calculate the accuracy as our performance metric
    accuracy = metrics.accuracy_score(val_y, pred_y)
    print("Accuracy: ", accuracy)

    # Get and print the classification report
    class_report = classification_report(val_y, pred_y)
    print("\nClassification Report:\n", class_report)

    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn import metrics

    # Assuming you have 'val_y' and 'pred_y' defined
    confusion = metrics.confusion_matrix(val_y, pred_y)

    # Display the confusion matrix
    print(f"Confusion matrix:\n{confusion}")

    # Plot the confusion matrix as a heatmap
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted Negative", "Predicted Positive"],
                yticklabels=["Actual Negative", "Actual Positive"])

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

     


    # SVM 
    svm_model = SVC()
    svm_model.fit(train_X, train_y)
    svm_predictions = svm_model.predict(val_X)

    # SVM metrics
    svm_accuracy = accuracy_score(val_y, svm_predictions)
    svm_f1 = f1_score(val_y, svm_predictions, average='weighted')  
    svm_r2 = r2_score(val_y, svm_predictions)
    svm_matrix = confusion_matrix(val_y, svm_predictions)
    svm_report = classification_report(val_y, svm_predictions)

    # KNN 
    knn_model = KNeighborsClassifier() 
    knn_model.fit( train_X, train_y) 
    knn_predictions = knn_model.predict(val_X)  

    # KNN metrics
    knn_accuracy = accuracy_score(val_y, knn_predictions)
    knn_f1 = f1_score(val_y, knn_predictions, average='weighted')
    knn_r2 = r2_score(val_y, knn_predictions) 
    knn_matrix = confusion_matrix(val_y, knn_predictions)
    knn_report = classification_report(val_y, knn_predictions)

    print('SVM Accuracy:', svm_accuracy)
    print('KNN Accuracy:', knn_accuracy )


    # Normalizing by the true label counts to get rates
    print(f"\nNormalized confusion matrix:")
    for row in confusion:
        print(row / row.sum())

 

    

    return render(request,'users/ml.html',{'accuracy':accuracy,
                                            'svm_accuracy':svm_accuracy,
                                            'svm_matrix':svm_matrix,
                                            'knn_matrix':knn_matrix,
                                            'knn_accuracy':knn_accuracy,
                                            'knn_report':knn_report,
                                            'svm_report':svm_report,
                                            'confusion':confusion,
                                            'class_report':class_report,
                                        })





def predictTrustWorthy(request):
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    if request.method == 'POST':
        # Extracting data from the POST request
        age = request.POST.get("age")
        gender = request.POST.get("gender")
        course_level = request.POST.get("course_level")
        academic_grade = request.POST.get("academic_year")
        midterm_grade = request.POST.get("midterm_grade")
        highschool_grade = request.POST.get("highschool_grade")
        Absence = request.POST.get("Absence")
 
 
        # Loading the dataset
        data = pd.read_csv(r'D:\Personal Projects\StudentPerformance\media\student.csv')
        
        data['grade'] = data['grade'].replace(['>90','<90'],[1,0])

        from sklearn.preprocessing import LabelEncoder                

        lb = LabelEncoder()
        data['gender']= lb.fit_transform(data['gender'])
        data['course_level'] = lb.fit_transform(data['course_level'])
        gender_encoded=lb.fit_transform([gender])[0]
        graduate_encoded = lb.fit_transform([course_level])[0]

        y = data['grade']
        X = data.drop('grade', axis=1)

        # Split the data into training and validation sets
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=7)

        model = RandomForestClassifier(random_state=7, n_estimators=100)
        model.fit(train_X, train_y)

        scaler = StandardScaler()
        user_input =  [age,gender_encoded,graduate_encoded, academic_grade, midterm_grade, highschool_grade, Absence] 
        print(user_input)

        # Predict classes given the validation features
        pred_y = model.predict([user_input])
        print('RandomForestClassifier:',pred_y)  

        # SVM 
        svm_model = SVC()
        svm_model.fit(train_X, train_y)
        svm_predictions = svm_model.predict([user_input])
         
        print('Support Vector Machine:',svm_predictions)  
 
        # Converting the prediction to 'yes' or 'no'
        if pred_y[0] == 1:
            msg = 'A Grade(>90)'
        else:
            msg = 'Below A Grade(<90)'

        return render(request, "users/predictionForm.html", {'msg':msg})
    else:
        return render(request, "users/predictionForm.html", {})