from flask import Flask
from flask_restful import Api,request
from flask import Response
import os
import json
import requests
from ai_api_client_sdk.ai_api_v2_client import AIAPIV2Client
app = Flask(__name__)
api = Api(app)
destval = {}


def get_ans(test_input):
    aic_service_key_path = 'keys.json'
    
    # Loads the service key file
    with open(aic_service_key_path) as ask:
        aic_service_key = json.load(ask)

# Creating an SAP AI API client instance
    ai_api_client = AIAPIV2Client(
    base_url = aic_service_key["serviceurls"]["AI_API_URL"] + "/v2", # The present SAP AI API version is 2
    auth_url=  aic_service_key["url"] + "/oauth/token",
    client_id = aic_service_key['clientid'],
    client_secret = aic_service_key['clientsecret'])
    print("here1")
	
    endpoint ="https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d73a38348d9d2eef/v1/check"
    headers = {"Authorization": ai_api_client.rest_client.get_token(),
           "ai-resource-group": "default",
           "Content-Type": "application/json"}
    response = requests.post(endpoint, headers=headers, json=test_input)
    try:
    
        destval = json.dumps(response.json())
    except:
        destval ="no record found"
        
    return destval

'''def register_record(register_input):
    aic_service_key_path = 'keys.json'
    
    # Loads the service key file
    with open(aic_service_key_path) as ask:
        aic_service_key = json.load(ask)

# Creating an SAP AI API client instance
    ai_api_client = AIAPIV2Client(
    base_url = aic_service_key["serviceurls"]["AI_API_URL"] + "/v2", # The present SAP AI API version is 2
    auth_url=  aic_service_key["url"] + "/oauth/token",
    client_id = aic_service_key['clientid'],
    client_secret = aic_service_key['clientsecret'])
    print("here1")
	
    endpoint ="https://api.ai.prod.eu-central-1.aws.ml.hana.ondemand.com/v2/inference/deployments/d73a38348d9d2eef/v1/register"
    headers = {"Authorization": ai_api_client.rest_client.get_token(),
           "ai-resource-group": "default",
           "Content-Type": "application/json"}
    response = requests.post(endpoint, headers=headers, json=register_input)
    try:
    
        destval = json.dumps(response.json())
    except:
        destval ="regitration failed"
        
    return destval'''








@app.route('/fetch',methods=['GET','POST'])
def fetch():
    if request.method=='POST':
        value =request.json['result']
        print("here3")
        answer = get_ans(value)
        res1 =Response(answer)
        res1.headers["Content-Type"]="application/json"
        res1.headers["Access-Control-Allow-Origin"]="*"
        return(res1)
 
'''@app.route('/register',methods = ['GET','POST'])
def register():
    if request.method=='POST':
        value = request.json['result']
        ans = register_record(value)
        res = Response(ans)
        res.headers["Content-Type"]="application/json"
        res.headers["Access-Control-Allow-Origin"]="*"
        return(res)'''
        
   

port =os.getenv('PORT',5000)
if __name__ == '__main__':
    app.run(host ="0.0.0.0",port=port)


