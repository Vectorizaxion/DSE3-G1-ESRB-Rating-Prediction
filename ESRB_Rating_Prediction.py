from flask import Flask, request
from flask_cors import CORS,cross_origin
import joblib
import numpy as np
import os
app = Flask(__name__)
CORS(app)

@app.route('/ESRB-Rating-Prediction',methods=['POST'])
@cross_origin()
def ESRB_predict_rating():
    model = joblib.load('ESRB_RF.model')
    req = request.values['param']
    input = np.array(req.split(','),dtype=np.float32).reshape(1,-1)
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

    Output = {  'Predicted Rating': predict_value,
                'Confident' : round(confident_value[0],3)
            }
        
    return  Output
    #predict_value + '\n Confidence : ' + str(round(confident_value[0],3))



if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host = '0.0.0.0', port = port)