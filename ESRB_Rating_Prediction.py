from flask import Flask, request
from flask_cors import CORS,cross_origin
from flask_restful import Resource ,Api ,reqparse
import joblib
import numpy as np
import os
import requests
app = Flask(__name__)
CORS(app)

@app.route('/ESRB-Rating-Prediction',methods=['POST'])
@cross_origin()
def ESRB_predict_rating():

    model = joblib.load('ESRB_RF.model')
    req = reqparse.RequestParser()
    req.add_argument('GameTitle',type=str,help='Input Game title')
    req.add_argument('AlcoholReference',type=int) 
    req.add_argument('AnimatedBlood',type=int)
    req.add_argument('Blood',type=int)
    req.add_argument('BloodandGore',type=int)
    req.add_argument('CartoonViolence',type=int)
    req.add_argument('CrudeHumor',type=int)
    req.add_argument('DrugReference',type=int)
    req.add_argument('FantasyViolence',type=int)
    req.add_argument('IntenseViolence',type=int)
    req.add_argument('Language',type=int)
    req.add_argument('Lyrics',type=int)
    req.add_argument('MatureHumor',type=int)
    req.add_argument('MildBlood',type=int)
    req.add_argument('MildCartoonViolence',type=int)
    req.add_argument('MildFantasyViolence',type=int)
    req.add_argument('MildLanguage',type=int)
    req.add_argument('MildLyrics',type=int)
    req.add_argument('MildSuggestiveThemes',type=int)
    req.add_argument('MildViolence',type=int)
    req.add_argument('NoDescriptors',type=int)
    req.add_argument('Nudity',type=int)
    req.add_argument('PartialNudity',type=int)
    req.add_argument('SexualContent',type=int)
    req.add_argument('SexualThemes',type=int)
    req.add_argument('SimulatedGambling',type=int)
    req.add_argument('StrongLanguage',type=int)
    req.add_argument('StrongSexualContent',type=int)
    req.add_argument('SuggestiveThemes',type=int)
    req.add_argument('UseofAlcohol',type=int)
    req.add_argument('UseofDrugsandAlcohol',type=int)
    req.add_argument('Violence',type=int)

    dictp = req.parse_args()
    param_list = list(dictp.values())

    
    Game_Title = str(param_list[0])

    features =  str(param_list[1:]).strip('[').strip(']')
    input = np.array(features.split(','),dtype=np.float32).reshape(1,-1)
    predict_target = model.predict(input)
    confident_value = model.predict_proba(input)
    confident_value = confident_value[0][predict_target]
    predict_value = ''

    if predict_target == 0 :
         predict_value = 'Everyone'
    elif predict_target == 1 :
         predict_value = 'Everyone 10+'
    elif predict_target == 2 :
         predict_value = 'Teen'
    else:
         predict_value = 'Mature 17+'

    Output = {  'Game Title : ' :Game_Title,
                'Rating : ': predict_value,
                'Confident' : round(confident_value[0],3)
             }
    return  Output



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host = '0.0.0.0', port = port)