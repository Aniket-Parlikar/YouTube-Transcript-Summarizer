from youtube_transcript_api import YouTubeTranscriptApi as yt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
tokenizer = AutoTokenizer.from_pretrained("t5-base")

def obtain_Transcript(video_id):
    transcript=' '
    transcript_list= yt.get_transcript(video_id)
    for elem in transcript_list:
        transcript+=elem['text']
    
    return transcript

def summarize_transcript(video_id):
    transcript=obtain_Transcript(video_id)
    inputs=tokenizer("summarize: " + transcript.strip(), return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(
    inputs["input_ids"], max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True
    )
    tokenizer.decode(outputs[0],format='utf-8')


summarize_transcript("WoDH-Xc0dGw")