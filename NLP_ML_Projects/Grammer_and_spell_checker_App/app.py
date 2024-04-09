from flask import Flask,request,render_template
from model import SpellCheckerModule

app=Flask(__name__)
spell_checker_module=SpellCheckerModule()
indexpath="D:\\NLP\\NLP_ML_Projects\\Grammer_and_spell_checker_App\\Templates\\index.html"
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/spell',methods=["POST","GET"])
def spell():
    if request.method=="POST":
        text=request.form['text']
        corrected_text=spell_checker_module.correct_spell(text)
        length_corrected_words=len(corrected_text)
        return render_template('index.html',corrected_text=corrected_text,length_corrected_words=length_corrected_words)

if __name__ =="__main__":
    app.run(debug=True)
