from flask import Flask , render_template ,request,redirect,render_template
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json
import plotly
import plotly.express as px

from werkzeug.datastructures import RequestCacheControl
from werkzeug.utils import redirect

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI']="sqlite:///todo.db"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
db=SQLAlchemy(app)

class Todo(db.Model):
    sno=db.Column(db.Integer,primary_key=True)
    title=db.Column(db.String(200),nullable=False)
    desc=db.Column(db.String(1000),nullable=True)
    date_created=db.Column(db.DateTime,default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"{self.sno} - {self.title}"


@app.route("/",methods=['GET','POST'])

def model():
    if request.method=="POST": 
        model_c=request.form['model']
        feature=request.form.getlist('feature_checkbox')
        feature.append("Open")
        length=int(request.form['Length'])
        
        if(model_c=='LSTM'):
            
            import numpy as np
            import pandas as pd
            import plotly.offline as py
            import plotly.graph_objs as go
            import yfinance  as yf
            from sklearn.preprocessing import MinMaxScaler

            company=request.form['company']

            data = yf.Ticker(company)
            
            df1 = data.history(period="5y")
            num_index = np.arange(0, len(df1))
            df1 =  df1.reset_index()
            df1 = df1.set_index(num_index)
            
            

            

            

            df1['Date']=pd.to_datetime(df1['Date'])


            X=df1[feature]
            print(X)
            Y=df1[['Close']]
            print(Y)
            dates = df1['Date']
            #length=30

            actual_close = []

            training_set = X.iloc[:1000].values
            training_set_close = Y.iloc[:1000].values

            test_set = X.iloc[1000-length:].values
            test_set_close = Y.iloc[1000-length:].values

            test_dates = dates[1000:]

            test_length = len(test_dates)

            maxClose = max(test_set_close)
            minClose = min(test_set_close)


            sc = MinMaxScaler(feature_range = (0, 1))


            training_set_scaled = sc.fit_transform(training_set)
            training_set_scaled_close = sc.fit_transform(training_set_close)
            test_set_scaled=sc.fit_transform(test_set)
            test_set_scaled_close=sc.fit_transform(test_set_close)

            X_train = []
            y_train_close = []

            for i in range(length, len(training_set)):
                X_train.append(training_set_scaled[i-length:i, :])
                y_train_close.append(training_set_scaled_close[i, 0])
                
            X_train, y_train_close = np.array(X_train), np.array(y_train_close)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))


            X_test = []
            y_test_close = []
            for i in range(length, len(test_set)):
                X_test.append(test_set_scaled[i-length:i, :])
                y_test_close.append(test_set_scaled_close[i, 0])
                actual_close.append(test_set_close[i,0])
                
            X_test, y_test_close = np.array(X_test), np.array(y_test_close)
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))


            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import Dense, LSTM, Dropout
            model = Sequential()
            model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
            model.add(Dropout(0.2))
            model.add(LSTM(units = 50, return_sequences = True))
            model.add(Dropout(0.2))
            model.add(LSTM(units = 50, return_sequences = True))
            model.add(Dropout(0.2))
            model.add(LSTM(units = 50))
            model.add(Dropout(0.2))
            model.add(Dense(units = 1))

            model.compile(optimizer = 'adam', loss = 'mean_squared_error',metrics=['accuracy'])

            history = model.fit(X_train, y_train_close, validation_data=(X_test,y_test_close),epochs = 10, batch_size = 32)

            pred=model.predict(X_test)


            y_test_close = y_test_close.reshape(-1, 1)

            sc = MinMaxScaler(feature_range = (minClose, maxClose))

            pred_scaled = sc.fit_transform(pred)
            y_test_scaled_close = sc.fit_transform(y_test_close)

            test=pd.DataFrame(columns=['test','pred'])
            test['test']=y_test_scaled_close.flatten()
            test['pred']=pred_scaled.flatten()

            data = []
            data.append(go.Scatter(x = test_dates, y = test['pred'].values,name = "Prediction"))
            #data.append(go.Scatter(x = test_dates, y = test['test'].values,name = "Actual" ))
            data.append(go.Scatter(x = test_dates, y = actual_close,name = "ORIGINAL" ))
            
            layout = go.Layout(dict(title = "Predicted and Actual Closing prices of  " + company + "  asset",
                            xaxis = dict(title = 'Year'),
                            yaxis = dict(title = 'Price'),
                            ),legend=dict(
                            orientation="h"))

            #fig  = go.Figure(data=data, layout=layout)
            py.plot(dict(data=data, layout=layout), filename='LSTM_'+ company +'_.html')
            
            
            #fig.show()
            return render_template('index.html')
            
        if(model_c=='Linear'):
            import numpy as np
            import pandas as pd
            import plotly.offline as py
            import plotly.graph_objs as go
            import yfinance  as yf
            from sklearn.preprocessing import MinMaxScaler

            company=request.form['company']

            data = yf.Ticker(company)
            
            df1 = data.history(period="5y")
            num_index = np.arange(0, len(df1))
            df1 =  df1.reset_index()
            df1 = df1.set_index(num_index)

            df1['Date']=pd.to_datetime(df1['Date'])
            feature.append("Date")
            X=df1[feature]
           
            y=df1['Close']

            XX_train = X[:1000]
            XX_test = X[1000:]

            y_train = y[:1000]
            y_test = y[1000:]

            feature.remove("Date")
            X_train = XX_train[feature]
            X_test = XX_test[feature]
            # print(type(X_train))
            # print(X_train.shape)
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test =scaler.transform(X_test)

            from sklearn.linear_model import LinearRegression
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            from sklearn import set_config
            model = LinearRegression()
            model.fit(X_train,y_train)
            set_config(display='diagram')
            pred=model.predict(X_test)
            sc=np.round(model.score(X_test, y_test),2) * 100
            r2=np.round(r2_score(y_test,pred),2)
            mse=np.round(mean_squared_error(y_test,pred),2)
            mae=np.round(mean_squared_error(y_test,pred),2)

            pred_df = pd.DataFrame(pred,y_test.index,['Prediction'])
            y_test = pd.DataFrame(y_test,y_test.index,['Close'])

            XX_test = XX_test.join(pred_df)
            XX_test = XX_test.join(y_test)

            data = []
            data.append(go.Scatter(x = XX_test['Date'], y = XX_test['Prediction'].values,name = "Prediction"))
            data.append(go.Scatter(x = XX_test['Date'], y = XX_test['Close'].values,name = "Actual" ))
            layout = go.Layout(dict(title = "Predicted and Actual Closing prices of  "+company+"  asset",
                            xaxis = dict(title = 'Year'),
                            yaxis = dict(title = 'Price (USD)'),
                            ),legend=dict(
                            orientation="h"))

            
            py.plot(dict(data=data, layout=layout), filename='Linear_Regrssion_'+company+'.html')

            return render_template('index.html')
                    
    return render_template('index.html')



    
if __name__=="__main__":
    app.run(debug=True)