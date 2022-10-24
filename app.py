import pandas as pd
import numpy as np
from flask import Flask
from flask_restful import Api,request
from flask import Response
from csv import writer
from scipy.sparse import csr_matrix
import sparse_dot_topn.sparse_dot_topn as ct
import re
import ftfy
from ftfy import fix_text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import os

app=Flask(__name__)
api = Api(app)

#df = pd.read_csv("newdata2.csv")
def ngrams(string,n=3):
    string = str(string)
    string =fix_text(string)
    string=string.encode("ascii",errors ="ignore").decode()
    string = string = string.lower()
    chars_to_remove = [")","(",".","|","[","]","{","}","'"]
    rx = '[' + re.escape(''.join(chars_to_remove)) + ']'
    string = re.sub(rx, '', string)
    string = string.replace('&', 'and')
    string = string.replace(',', ' ')
    string = string.replace('-', ' ') 
    string = string.replace('@', ' ')
    string =string.title()
    string = re.sub(' +',' ',string).strip()
    string = ' '+ string +' '
    string = re.sub(r'[,-./]|\sBD',r'', string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return [''.join(ngram) for ngram in ngrams]
	


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape
 
    idx_dtype = np.int32
 
    nnz_max = M*ntop
 
    indptr = np.zeros(M+1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)
    ct.sparse_dot_topn(
        M, N, np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr, indices, data)

    return csr_matrix((data,indices,indptr),shape=(M,N))
	
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def similarity_checking(names,string):
    #org_name_clean = df['Name'].unique()
    global vectorizer

    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, lowercase=False)
    tfidf = vectorizer.fit_transform(names)
    from sklearn.neighbors import NearestNeighbors

    def getNearestN(query):
      queryTFIDF_ = vectorizer.transform(query)
      distances, indices = nbrs.kneighbors(queryTFIDF_)
      return distances, indices


    nbrs = NearestNeighbors(n_neighbors=1, n_jobs=-1).fit(tfidf)
       

    
    distances, indices = getNearestN(string)
    string =list(string)
    

    matches = []
    for i,j in enumerate(indices):
      temp = [round(distances[i][0],2), names[j][0],string[i]]
      matches.append(temp)

    matches = pd.DataFrame(matches, columns=['Match confidence','Matched name','Origional name'])
    return matches


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



	
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def most_common(lst):
    return max(set(lst), key=lst.count)
	
	
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def dedups(dict2):
    df = pd.read_csv("newdata.csv")
    #print("point1")
    X = dict2.values()
    X=list(X)
    test =pd.DataFrame(dict2,index=[0])

    Fnames =np.array(list(df['first_name']))
    Lnames =np.array(list(df['last_name']))
    Gender=np.array(list(df['gender']))
    email=np.array(list(df['email']))
    dob =np.array(list(df["date of birth"]))
    Phno = np.array(list(df["phone number"]))
    address = np.array(list(df["full address"]))
    
    Fname_string=[]
    Lname_string=[]
    Address_string=[]
    Email_string=[]
    dob_string=[]
    Phno_string=[]
    Gender_string=[]

   
    
    Fname_string.append(X[0]) #test["Fname"]
    Fname_string1 = set(Fname_string)
    Lname_string.append((X[1] ))  #test["Lname"]
    Lname_string1 =set(Lname_string) 
    dob_string.append(X[2]) #test["Dob"]
    dob_string1 = set(dob_string)
    Phno_string.append(X[3]) #test["Ph no"]
    Phno_string1 = set(Phno_string)
    Address_string.append(X[4]) #test["Address"]
    Address_string1 = set(Address_string) 
    Gender_string.append(X[5]) #test["bank details"]
    Gender_string1 = set(Gender)
    Email_string.append(X[6])  #test["Email"]
    Email_string1 = set(Email_string)
    
    Fname_confidence =  similarity_checking(Fnames,Fname_string1)
    Lname_confidence = similarity_checking(Lnames,Lname_string1)
    Dob_confidence = similarity_checking(dob,dob_string1)
    Phno_confidence = similarity_checking(Phno,Phno_string1)
    Address_confidence = similarity_checking(address,Address_string1)
    Gender_confidence = similarity_checking(Gender,Gender_string1)
    Email_confidence = similarity_checking(email,Email_string1)

   

    ca=[Fname_confidence,Lname_confidence,Dob_confidence,Phno_confidence,Address_confidence,Email_confidence]
    cj=['first_name', 'last_name', 'date of birth','phone number', 'full address', 'email']
    def get_index(ca,cf_index):
        j=0
        for i in ca:
            get_ans=i["Matched name"][0]
            qw=df[df[cj[j]]==get_ans].index.values
            
            j+=1
            if i["Match confidence"][0] < 1 :
                cf_index.append(qw[0])

        return cf_index

    cf_index=[]
    
    cf_index = get_index(ca,cf_index)
    if len(cf_index) == 0:
        common_index=0
    else:
    
        common_index=most_common(cf_index)
    

    if common_index ==0:
        check=np.array(df.loc[0:1])
    else:
        check=np.array(df.loc[common_index-1:common_index])
        
    answer1 = similarity_checking(check,np.array(test))
    if answer1["Match confidence"].values > 1:
        return "No matching records found"
    else:
        return answer1.to_json()
	
	
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++	

def appender(result):
    neer=list(result.values())
    with open('newdata.csv', 'a') as filer:
        writing = writer(filer)
        writing.writerow(neer)
        filer.close
        return "Successfully Added"



@app.route('/v1/check',methods=['GET','POST'])
def check():
    if request.method =='POST' :
        value =request.json["result"]
        print("point0")
        print(value)
        result= dedups(value)
        res =Response(result)
        res.headers["Content-Type"]="application/json"
        res.headers["Access-Control-Allow-Origin"]="*"
        return res
	
@app.route('/v1/register',methods=['GET','POST'])
def register():
    if request.method=='POST':
        value =request.json['result']
        answer = appender(value)
        res1 =Response(answer)
        res1.headers["Content-Type"]="application/json"
        res1.headers["Access-Control-Allow-Origin"]="*"
        return(res1)

port =os.getenv('PORT',5000)
if __name__ == '__main__':
    app.run(host ="0.0.0.0",port=port)


    
 

