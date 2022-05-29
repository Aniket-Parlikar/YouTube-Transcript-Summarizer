from flask import Flask,jsonify,request
import datetime
from youtube_transcript_api import YouTubeTranscriptApi as yt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from yt_test import obtain_Transcript

#Initializing the model and tokenizers for summarizing the text
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

#define a variable to hold your app name
app = Flask(__name__)

#Define your resource endpoints
@app.route('/')
def index_page():
    return 'Hello World!'

#Define a placeholder function for the boilerplate
@app.route('/time',methods=['GET'])
def get_time():
    return str(datetime.datetime.now())

#Define the function to summarize the text
@app.route('/api/summarize',methods=['GET'])
def summarize_transcript():
    #Accessing the url name
    youtube_url=request.args.get('youtube_url','')
    #https://www.youtube.com/watch?v=8TW_C9UVQ1I
    video_id=youtube_url.split('=')[-1]
    try:
        transcript=obtain_Transcript(video_id)
        inputs=tokenizer("summarize: " + transcript, return_tensors="pt", max_length=512, truncation=True)
        outputs = model.generate(inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
        summarized_text=tokenizer.decode(outputs[0],format='utf-8')
    except:
        summarized_text="Sorry!You have entered an incorrect video url"

    op={'video_id':video_id,'summarization':summarized_text}
    return jsonify(op)

##server the app when ready
if __name__=='__main__':
    app.run()