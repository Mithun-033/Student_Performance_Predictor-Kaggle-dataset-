import pandas as pd
import torch
import numpy as np

df=pd.read_csv('StudentPerformanceFactors.csv')
df["Parental_Involvement"]=df["Parental_Involvement"].map({'Low':1,'Medium':2,'High':3})
df["Access_to_Resources"]=df["Access_to_Resources"].map({'Low':1,'Medium':2,'High':3})
df["Extracurricular_Activities"]=df["Extracurricular_Activities"].map({"Yes":1,"No":0})
df["Motivation_Level"]=df["Motivation_Level"].map({'Low':1,'Medium':2,'High':3})
df["Internet_Access"]=df["Internet_Access"].map({"Yes":2,"No":1})
df["Family_Income"]=df["Family_Income"].map({'Low':1,'Medium':2,'High':3})
df["Teacher_Quality"]=df["Teacher_Quality"].map({'Low':1,'Medium':2,'High':3})
df["School_Type"]=df["School_Type"].map({'Public':1,'Private':2})
df["Peer_Influence"]=df["Peer_Influence"].map({'Positive':1,'Neutral':2,'Negative':3})
df["Learning_Disabilities"]=df["Learning_Disabilities"].map({"Yes":2,"No":1})
df["Parental_Education_Level"]=df["Parental_Education_Level"].fillna(df["Parental_Education_Level"].mode()[0])
df["Parental_Education_Level"]=df["Parental_Education_Level"].map({'High School':1,'College':2,'Postgraduate':3})
df["Distance_from_Home"]=df["Distance_from_Home"].fillna(df["Distance_from_Home"].mode()[0])
df["Distance_from_Home"]=df["Distance_from_Home"].map({'Near':1,'Moderate':2,'Far':3})
df["Gender"]=df["Gender"].map({"Male":2,"Female":1})

X=np.array(df.values[:,:-1])
Y=np.array(df.values[:,-1])
#copying the weights and biases and also the normalised params
c=torch.load('Student_Performance_8.pth',weights_only=False)
w1,b1,w2,b2=c["w1"],c["b1"],c["w2"],c["b2"]
X_mean=c["X_mean"]
X_std=c["X_std"]
Y_mean=c["Y_mean"]
Y_std=c["Y_std"]


X=(X-X_mean)/X_std
Y_norm=(Y-Y_mean)/Y_std

split=int(len(X)*0.7)
X_test=torch.tensor(X[split:]).float()
Y_test=torch.tensor(Y_norm[split:]).float().view(-1,1)
Y_actual=torch.tensor(Y[split:]).float().view(-1,1)
#calculating predictions  
with torch.no_grad():
    z1=X_test@w1+b1
    a1=torch.relu(z1)
    y_pred_norm=a1@w2+b2
    y_pred=y_pred_norm*Y_std+Y_mean
#checking prediction sccuracy
acc=[]
for i in range(len(Y_actual)):
    n=min(abs(Y_actual[i].item()),abs(y_pred[i].item()))
    d=max(abs(Y_actual[i].item()),abs(y_pred[i].item()))
    acc.append(n/d)
final_acc=sum(acc)/len(acc)

print(f"Final Test Accuracy: {final_acc*100:.2f}%")
for i in range(5):
    print(Y_actual[i].item(),y_pred[i].item())
