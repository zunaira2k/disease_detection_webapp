from flask import Flask, render_template, request
app = Flask(__name__)
import pickle


# open a file, where you stored a pikled data
file = open('model.pkl','rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        fever = int(myDict['fever'])
        age = int(myDict['age'])
        bodyPain = int(myDict['bodyPain'])
        runnyNose = int(myDict['runnyNose'])
        tired = int(myDict['tired'])
        diarreha = int(myDict['diarreha'])
        sore_throat = int(myDict['sore_throat'])
        cough = int(myDict['cough'])
        diffBreath = int(myDict['diffBreath'])
    

        # Code for inference
        inputFeatures = [fever, bodyPain, age, runnyNose, diffBreath, tired, diarreha, sore_throat, cough ]
        infProb =clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html', inf=round(infProb*100))
    return render_template('index.html')
    # return 'Hello, World!' + str(infProb)


if __name__ == "__main__":
    app.run(debug=True)
