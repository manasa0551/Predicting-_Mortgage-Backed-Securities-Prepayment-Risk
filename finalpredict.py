import pandas as pd
import pickle
import app
from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.pipeline import Pipeline as pipe
import mgzip

# Credit Grade     
def creditgrade(Credit):
    if Credit<300:
        v=1
    elif Credit >=300 and Credit< 500:
        v=2
    elif Credit >=500 and Credit <= 750:
        v=3
    else:
        v=4
    
    if v==1:
        s="Very Poor"
    elif v==2:
        s="Fair"
    elif v==3:
        s="Good"
    elif v==4:
        s="Excellent"
    return (v, s)


# Loan Amount
def emi(p, r, t):        
    r = r/(12*100)  
    t = t*12
    emi = (p*r) * (1+r)**t/(((1+r)**t)-1)
    totalloanamount = round(emi * 360)
    totalinterest=totalloanamount-p
    return (round(emi),totalloanamount,totalinterest)        

# DTI Ratio 
def dtigrade(dtiratio):
    if dtiratio<40:
        return 'Chance of Prepayment of Loan is Too high'
    else:
        return 'Chance Prepayment of Loan is less'

# New Tenur
def tenure(emiamount,rateofinterest,loanamount,monthsinrepayment,lt):
    rateofinterest = rateofinterest/100
    iterestmonthlywise=loanamount * rateofinterest/12
    prepayment=emiamount-iterestmonthlywise
    removeingtenure=int(prepayment/monthsinrepayment)
    principleamount=loanamount-prepayment
    oldloantenure=lt*12
    newtenure=oldloantenure-monthsinrepayment-removeingtenure
    print("principalamount: ", principleamount)
    return round(newtenure)   

# ROI
def roidata(totalinterest,totalloanamount):
    roi= (totalinterest/ totalloanamount)*100     
    return round(roi)

# Predit Modle
def pipeline_output(X):
    with mgzip.open('classifierPipe', 'rb') as f:
        classifier_pipeline = pickle.load(f)

    #classifier_pipeline = pickle.load(open('classifier_Pipe.pkl','rb'))
    classifieroutput=classifier_pipeline.predict(X)
    regressor_pipeline = pickle.load(open('Regression_pipe.pkl','rb'))
    regressionoutput=regressor_pipeline.predict(X) 
    print("Deliquency Status: ",classifieroutput)
    print("Regressionoutput: ", regressionoutput)
    regressionone=round(regressionoutput[0][0])
    regressiontwo=round(regressionoutput[0][1])
    if classifieroutput[0]==0:    
        classout=classifieroutput[0]        
        print("Not Deliquent")
    else:          
        classout=classifieroutput[0]  
        print("Deliquent")      
    return (classout,regressionoutput,regressionone,regressiontwo)

# Output Print
def result(creditrange,emiamount,totalloanamount,totalinterest,dtiratiograde,newtenure,roi):
    print("Credit Score Range: ",creditrange)
    print(" Monthly Emi Amount: ", emiamount)
    print("Total LoanAmount to be Paid: ", float(totalloanamount))
    print("Total Interest to be Paid: ", totalinterest)
    print("DTI Ratiograde: ", dtiratiograde)
    print("Newtenure(IN MONTHS): ", newtenure)
    print("Retrun on Investment(ROI): ", roi)
    return True






