from transformers import pipeline
import os

# ==============================
# PREPARE SAMPLE INPUT FILES
# ==============================

# Create a sample text file
if not os.path.exists("sample_text.txt"):
    with open("sample_text.txt", "w") as f:
        f.write("""Mark Elliot Zuckerberg zkrbr born May 14 1984 is an American businessman who cofounded the social media service Facebook and its parent company Meta Platforms of which he is the chairman chief executive officer and controlling shareholder Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy
                Born in White Plains New York Zuckerberg briefly attended Harvard College where he launched Facebook in February 2004 with his roommates Eduardo Saverin Andrew McCollum Dustin Moskovitz and Chris Hughes Zuckerberg took the company public in May 2012 with majority shares He became the worlds youngest selfmade billionairea in 2008 at age 23 and has consistently ranked among the worlds wealthiest individuals According to Forbes Zuckerbergs estimated net worth stood at US2212 billion as of May 2025 making him the secondrichest individual in the world2
                Zuckerberg has used his funds to organize multiple large donations including the establishment of the Chan Zuckerberg Initiative A film depicting Zuckerbergs early career legal troubles and initial success with Facebook The Social Network was released in 2010 and won multiple Academy Awards His prominence and fast rise in the technology industry has prompted political and legal attention"""
                )

# Generate a small audio sample using gTTS (Google Text-to-Speech)
try:
    from gtts import gTTS
    if not os.path.exists("speech.wav"):
        tts = gTTS("Mark Elliot Zuckerberg zkrbr born May 14 1984 is an American businessman who cofounded the social media service Facebook and its parent company Meta Platforms of which he is the chairman chief executive officer and controlling shareholder Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy", lang="en")
        tts.save("speech.wav")
except ImportError:
    print("⚠️ gTTS not installed, skipping audio file creation. Install with: pip install gTTS")

# ==============================
# 1. Sentiment Analysis
# ==============================
print("\n--- 1 - A: Sentiment Analysis ---")
sentiment_pipeline = pipeline("sentiment-analysis")
print(sentiment_pipeline("Hugging Face makes working with AI so much easier!"))

# ==============================
# 1. Sentiment Analysis - For production, Hugging Face recommends explicitly specifying the model + revision to avoid unexpected behavior
# if defaults change in the future.:
# ==============================
print("\n--- 1 - B: Sentiment Analysis --- explicitly specifying the model + revision")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
    revision="714eb0f"  # pin to the exact version you tested
    #,device=0  # 0 means first GPU
)
print(sentiment_pipeline("Hugging Face makes working with AI so much easier!"))

# ==============================
# 2. Named Entity Recognition (NER)
# ==============================
print("\n--- 2. Named Entity Recognition ---")
ner_pipeline = pipeline("ner", grouped_entities=True)
print(ner_pipeline("Mark Zuckerberg is the founder, chairman and CEO of Meta, which he originally founded as Facebook in 2004"))

# ==============================
# 3. Question Answering
# ==============================
print("\n--- Question Answering ---")
qa_pipeline = pipeline("question-answering")
context = "Hugging Face is a company based in New York and Paris. It is famous for transformers library."
print(qa_pipeline(question="Where is Hugging Face based?", context=context))

# ==============================
# 4. Summarization
# ==============================
print("\n--- Summarization ---")
summarizer = pipeline("summarization")
with open("sample_text.txt", "r") as f:
    text = f.read()
print(summarizer(text, max_length=40, min_length=10, do_sample=False))

# ==============================
# 5. Text Generation
# ==============================
print("\n--- Text Generation ---")
generator = pipeline("text-generation", model="gpt2")
print(generator("Once upon a time in AI world,", max_length=30, num_return_sequences=1))

# ==============================
# 6. Translation
# ==============================
print("\n--- Translation (EN → FR) ---")
translator = pipeline("translation_en_to_fr")
print(translator("thank you"))

# ==============================
# 7. Zero-Shot Classification
# ==============================
print("\n--- Zero-Shot Classification ---")
zero_shot = pipeline("zero-shot-classification")
print(zero_shot(
    "I love to play football on weekends.",
    candidate_labels=["sports", "politics", "technology"]
))

# ==============================
# 8. Automatic Speech Recognition (ASR)
# ==============================
print("\n--- Automatic Speech Recognition ---")
try:
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-base")
    if os.path.exists("speech.wav"):
        print(asr("speech.wav"))
    else:
        print("⚠️ No audio file available.")
except Exception as e:
    print(f"⚠️ ASR pipeline skipped (model too large or missing dependency): {e}")

# ==============================
# 9. Image Classification
# ==============================
print("\n--- Image Classification ---")
image_classifier = pipeline("image-classification")
print(image_classifier("https://hips.hearstapps.com/roa.h-cdn.co/assets/15/10/nrm_1425400062-aero817.jpg?crop=0.894xw:0.671xh;0,0.209xh&resize=640:*"))
