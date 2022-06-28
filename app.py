#pip install flask
#pip install flask_restful
#pip install sentence_transformers
#pip install spellchecker
#pip install textstat

from flask import Flask,request
from flask_restful import Resource, Api
from sentence_transformers import SentenceTransformer
from scipy.spatial import distance
from spellchecker import SpellChecker
import re
import textstat

spell = SpellChecker()
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

app = Flask(__name__)
api = Api(app)

class Match(Resource):
    def get(self):

        res = 0
        peak = 0
        matchData = []
        matchData.append(request.args.get("testData"))
        matchData.append(request.args.get("referData1"))
        matchData.append(request.args.get("referData2"))
        matchData.append(request.args.get("referData3"))
        matchData.append(request.args.get("referData4"))
        matchData.append(request.args.get("referData5"))
        matchData.append(request.args.get("referData6"))
        matchData.append(request.args.get("referData4"))
        matchData.append(request.args.get("referData8"))
        matchData.append(request.args.get("referData9"))
        matchData.append(request.args.get("referData10"))

        sentence_embeddings = model.encode(matchData)

        for i in range(1,11):
            temp = (1 - distance.cosine(sentence_embeddings[0], sentence_embeddings[i]))
            res = res + temp
            if(peak < temp):
                peak = temp
        
        return {'match': res/10 , 'peak' : peak},200

class Spell(Resource):
    def get(self):
        testData = request.args.get("testData")
        correctDict = {}
        for val in re.split(r'[^\w]', testData): 
            if not val:
                continue 
            misspelled = spell.unknown([val])
            
            if len(misspelled) > 0:
                misWord = misspelled.pop()
                corrected = spell.correction(misWord) 
                correctDict[misWord] = corrected
            else: 
                correctDict[val] = val
        
        # Parse out the typos 
        res = {k:v for k,v in correctDict.items() if k != v}

        return {'spellings':res},200

class Readability(Resource):
    def get(self):
        testData = request.args.get("testData")
        res1 = (textstat.flesch_reading_ease(testData))
        res2 = (textstat.lexicon_count(testData, removepunct=True))

        return {'readability' : res1 , 'wordCount' : res2},200

api.add_resource(Match, '/match')
api.add_resource(Spell, '/spell')
api.add_resource(Readability, '/readability')

if __name__ == '__main__':
    app.run(debug=False)  # run our Flask app
