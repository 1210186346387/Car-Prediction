import numpy as np
from collections import Counter
# Features:[engine_cc,horsepower,weight_kg,model,seat_capacity]
x=np.array([
    [1000,80,700,2007,4],
    [660,60,550,2019,4],
    [800,70,600,2015,4],
    [1300,120,1000,2018,5],
    [1000,65,800,2005,4],
    [1500,150,1100,2018,5],
    [1800,200,1300,2025,5],
    [1300,150,1100,2018,5],
    [1500,150,1000,2020,5],
    [2500,350,1300,2008,5],
    [2000,220,1200,2006,5],
    [3000,370,1300,2018,7],
    [4000,205,1500,2022,7]
])
y=np.array([0,1,1,1,0,1,1,1,1,0,0,1,1])
x_train,y_train=x[:8],y[:8]
x_test,y_test=x[8:],y[8:]
def car_distance(a,b):
    return np.sqrt(np.sum((a-b)**2))
def knn_predict(x_train,y_train,x,k=3):
    distances=[car_distance(x,row)
for row in x_train]
    k_indecis=np.argsort(distances)[:k]
    k_labels=[y_train[i] for i in k_indecis]
    return Counter(k_labels).most_common(1)[0][0]
labels={0:'Bad Deal', 1:'Good Deal'}
print(f"{'Car':<8} {'Actual':<12} {'Predicted':<12} {'Match'}")
prediction=[]
for i,x in enumerate(x_test):
    pred=knn_predict(x_train,y_train,x)
    prediction.append(pred)
    match='OK' if pred == y_test[i] else "WRONG"
    print(f'Car{i+1} {labels[y_test[i]]:<12} {labels[pred]:<12} {match}')
    accuracy= np.mean(np.array(prediction)==y_test)
    print(f'Accuracy: {accuracy*100:.2f}%')
    print('\n New Car Prediction ')
    new_car=np.array([1600,190,1200,2013,5])
    result=knn_predict(x_train,y_train,new_car)
    print(f'Engine: 1600, HP: 190, Weight: 1200kg, Model: 2013, Seat_Capacity: 5')
    print(f'Prediction: {labels[result]}')