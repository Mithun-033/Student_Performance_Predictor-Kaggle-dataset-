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
df["Teacher_Quality"] = df["Teacher_Quality"].fillna("Medium")
df["Teacher_Quality"]=df["Teacher_Quality"].map({'Low':1,'Medium':2,'High':3})
df["School_Type"]=df["School_Type"].map({'Public':1,'Private':2})
df["Peer_Influence"]=df["Peer_Influence"].map({'Positive':1,'Neutral':2,'Negative':3})
df["Learning_Disabilities"]=df["Learning_Disabilities"].map({"Yes":2,"No":1})
df["Parental_Education_Level"]=df["Parental_Education_Level"].fillna("College")
df["Parental_Education_Level"]=df["Parental_Education_Level"].map({'High School':1,'College':2,'Postgraduate':3})
df["Distance_from_Home"]=df["Distance_from_Home"].fillna("Moderate")
df["Distance_from_Home"]=df["Distance_from_Home"].map({'Near':1,'Moderate':2,'Far':3})
df["Gender"]=df["Gender"].map({"Male":2,"Female":1})

X=np.array(df.values[:,:-1])
Y=np.array(df.values[:,-1])



split=int(len(X)*0.7)
X_train=X[:split]
Y_train=Y[:split]

X_mean=X_train.mean(axis=0)
X_std=X_train.std(axis=0)
Y_mean=Y_train.mean()
Y_std=Y_train.std()

X_train_norm=(X_train-X_mean)/X_std
Y_train_norm=(Y_train-Y_mean)/Y_std

X_input=torch.tensor(X_train_norm).float()
Y_input=torch.tensor(Y_train_norm).float().view(-1,1)

w1 = torch.randn(19,30) * (1/np.sqrt(19)); w1.requires_grad_()
b1 = torch.zeros(30, requires_grad=True)
w2 = torch.randn(30,1) * (1/np.sqrt(30)); w2.requires_grad_()
b2 = torch.zeros(1, requires_grad=True)

no_of_cycles=6000
batch_size=32
optimizer = torch.optim.Adam([w1, b1, w2, b2], lr=0.000003)
while no_of_cycles>0:
    for i in range(0,len(X_input),batch_size):
        X_batch=X_input[i:i+batch_size]
        Y_batch=Y_input[i:i+batch_size]
        
        z1=X_batch@w1+b1
        a1=torch.relu(z1)
        y_pred=a1@w2+b2
        
        loss=torch.mean((y_pred-Y_batch)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    no_of_cycles -= 1
    if no_of_cycles % 10 == 0:
        print(f"Cycle: {no_of_cycles}, Loss: {loss.item()}")

with torch.no_grad():
    z1=X_input@w1+b1
    a1=torch.relu(z1)
    y_train_pred=a1@w2+b2

acc=[]
for i in range(len(Y_input)):
    n=min(abs(Y_input[i].item()),abs(y_train_pred[i].item()))
    d=max(abs(Y_input[i].item()),abs(y_train_pred[i].item()))
    acc.append(n/d)
train_accuracy=sum(acc)/len(acc)
print("Train Accuracy:",train_accuracy*100)

torch.save({
    "w1":w1.detach(),
    "b1":b1.detach(),
    "w2":w2.detach(),
    "b2":b2.detach(),
    "X_mean":X_mean,
    "X_std":X_std,
    "Y_mean":Y_mean,
    "Y_std":Y_std
}, "Student_Performance_8.pth")