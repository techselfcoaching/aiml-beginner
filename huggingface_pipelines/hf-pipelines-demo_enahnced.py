from transformers import pipeline
import os
import warnings
warnings.filterwarnings("ignore")

print("🤗 Enhanced Hugging Face Pipelines Demo")
print("=" * 50)

# ==============================
# PREPARE SAMPLE INPUT FILES
# ==============================

# Create a sample text file
if not os.path.exists("sample_text.txt"):
    with open("sample_text.txt", "w") as f:
        f.write("""Mark Elliot Zuckerberg born May 14 1984 is an American businessman who cofounded the social media service Facebook and its parent company Meta Platforms of which he is the chairman chief executive officer and controlling shareholder. Zuckerberg has been the subject of multiple lawsuits regarding the creation and ownership of the website as well as issues such as user privacy.
                
Born in White Plains New York Zuckerberg briefly attended Harvard College where he launched Facebook in February 2004 with his roommates Eduardo Saverin Andrew McCollum Dustin Moskovitz and Chris Hughes. Zuckerberg took the company public in May 2012 with majority shares. He became the worlds youngest selfmade billionaire in 2008 at age 23 and has consistently ranked among the worlds wealthiest individuals. According to Forbes Zuckerbergs estimated net worth stood at US$221.2 billion as of May 2025 making him the second-richest individual in the world.
                
Zuckerberg has used his funds to organize multiple large donations including the establishment of the Chan Zuckerberg Initiative. A film depicting Zuckerbergs early career legal troubles and initial success with Facebook The Social Network was released in 2010 and won multiple Academy Awards. His prominence and fast rise in the technology industry has prompted political and legal attention.""")

# Generate a small audio sample using gTTS (Google Text-to-Speech)
try:
    from gtts import gTTS
    if not os.path.exists("speech.wav"):
        tts = gTTS("Hello world, this is a sample audio for speech recognition testing.", lang="en")
        tts.save("speech.wav")
        print("✅ Sample audio file created: speech.wav")
except ImportError:
    print("⚠️ gTTS not installed, skipping audio file creation. Install with: pip install gTTS")

# ==============================
# 1. Sentiment Analysis
# ==============================
print("\n--- 1-A: Sentiment Analysis (Default Model) ---")
try:
    sentiment_pipeline = pipeline("sentiment-analysis")
    result = sentiment_pipeline("Hugging Face makes working with AI so much easier!")
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n--- 1-B: Sentiment Analysis (Explicit Model) ---")
try:
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        revision="714eb0f"  # pin to the exact version you tested
        #,device=0  # 0 means first GPU
    )
    result = sentiment_pipeline("I'm feeling quite disappointed with this product.")
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 2. Named Entity Recognition (NER)
# ==============================
print("\n--- 2: Named Entity Recognition ---")
try:
    ner_pipeline = pipeline("ner", grouped_entities=True)
    result = ner_pipeline("Mark Zuckerberg is the founder, chairman and CEO of Meta, which he originally founded as Facebook in 2004")
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 3. Question Answering
# ==============================
print("\n--- 3: Question Answering ---")
try:
    qa_pipeline = pipeline("question-answering")
    context = "Hugging Face is a company based in New York and Paris. It is famous for transformers library and democratizing AI."
    result = qa_pipeline(question="Where is Hugging Face based?", context=context)
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 4. Summarization
# ==============================
print("\n--- 4: Summarization ---")
try:
    summarizer = pipeline("summarization")
    with open("sample_text.txt", "r") as f:
        text = f.read()
    result = summarizer(text, max_length=50, min_length=20, do_sample=False)
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 5. Text Generation
# ==============================
print("\n--- 5: Text Generation ---")
try:
    generator = pipeline("text-generation", model="gpt2")
    result = generator("Once upon a time in AI world,", max_length=50, num_return_sequences=1, do_sample=True, temperature=0.7)
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 6. Translation
# ==============================
print("\n--- 6: Translation (EN → FR) ---")
try:
    translator = pipeline("translation_en_to_fr")
    result = translator("Thank you for using Hugging Face transformers!")
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 7. Zero-Shot Classification
# ==============================
print("\n--- 7: Zero-Shot Classification ---")
try:
    zero_shot = pipeline("zero-shot-classification")
    result = zero_shot(
        "I love to play football on weekends and watch matches with friends.",
        candidate_labels=["sports", "politics", "technology", "entertainment", "health"]
    )
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 8. Automatic Speech Recognition (ASR)
# ==============================
print("\n--- 8: Automatic Speech Recognition ---")
try:
    asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny")  # Using tiny model for faster loading
    if os.path.exists("speech.wav"):
        result = asr("speech.wav")
        print(f"✅ Result: {result}")
    else:
        print("⚠️ No audio file available.")
except Exception as e:
    print(f"❌ ASR pipeline skipped (model too large or missing dependency): {e}")

# ==============================
# 9. Image Classification
# ==============================
print("\n--- 9: Image Classification ---")
try:
    image_classifier = pipeline("image-classification")
    result = image_classifier("https://hips.hearstapps.com/roa.h-cdn.co/assets/15/10/nrm_1425400062-aero817.jpg?crop=0.894xw:0.671xh;0,0.209xh&resize=640:*")
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 10. NEW: Fill-Mask (BERT-style)
# ==============================
print("\n--- 10: Fill-Mask ---")
try:
    fill_mask = pipeline("fill-mask")
    result = fill_mask("Hugging Face is creating a [MASK] that the community loves.")
    print(f"✅ Result: {result[:3]}")  # Show top 3 predictions
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 11. NEW: Feature Extraction (Embeddings)
# ==============================
print("\n--- 11: Feature Extraction (Embeddings) ---")
try:
    feature_extractor = pipeline("feature-extraction", model="distilbert-base-uncased")
    result = feature_extractor("Hugging Face is amazing!")
    print(f"✅ Result: Embedding shape: {len(result[0])} dimensions")
    print(f"    First 5 values: {result[0][:5]}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 12. NEW: Token Classification (NER Alternative)
# ==============================
print("\n--- 12: Token Classification ---")
try:
    token_classifier = pipeline("token-classification", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    result = token_classifier("Apple Inc. was founded by Steve Jobs in California.")
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 13. NEW: Text2Text Generation (T5-style)
# ==============================
print("\n--- 13: Text2Text Generation ---")
try:
    text2text = pipeline("text2text-generation", model="t5-small")
    result = text2text("translate English to German: How are you doing today?")
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 14. NEW: Object Detection
# ==============================
print("\n--- 14: Object Detection ---")
try:
    object_detector = pipeline("object-detection", model="facebook/detr-resnet-50")
    result = object_detector("https://hips.hearstapps.com/roa.h-cdn.co/assets/15/10/nrm_1425400062-aero817.jpg?crop=0.894xw:0.671xh;0,0.209xh&resize=640:*")
    print(f"✅ Result: Found {len(result)} objects")
    for obj in result[:3]:  # Show first 3 objects
        print(f"    {obj['label']}: {obj['score']:.3f}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 15. NEW: Depth Estimation
# ==============================
print("\n--- 15: Depth Estimation ---")
try:
    depth_estimator = pipeline("depth-estimation", model="Intel/dpt-large")
    result = depth_estimator("https://hips.hearstapps.com/roa.h-cdn.co/assets/15/10/nrm_1425400062-aero817.jpg?crop=0.894xw:0.671xh;0,0.209xh&resize=640:*")
    print(f"✅ Result: Depth map generated with shape: {result['depth'].shape if hasattr(result['depth'], 'shape') else 'N/A'}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 16. NEW: Table Question Answering
# ==============================
print("\n--- 16: Table Question Answering ---")
try:
    table_qa = pipeline("table-question-answering", model="google/tapas-base-finetuned-wtq")
    table = {
        "Company": ["Apple", "Google", "Microsoft", "Meta"],
        "Revenue (2023)": ["$383B", "$307B", "$211B", "$134B"],
        "Employees": ["164,000", "182,000", "221,000", "86,000"]
    }
    result = table_qa(table=table, query="Which company has the highest revenue?")
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

# ==============================
# 17. NEW: Visual Question Answering
# ==============================
print("\n--- 17: Visual Question Answering ---")
try:
    vqa = pipeline("visual-question-answering", model="dandelin/vilt-b32-finetuned-vqa")
    result = vqa(
        image="https://hips.hearstapps.com/roa.h-cdn.co/assets/15/10/nrm_1425400062-aero817.jpg?crop=0.894xw:0.671xh;0,0.209xh&resize=640:*",
        question="What color is this vehicle?"
    )
    print(f"✅ Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 50)
print("🎉 Demo completed! Check the results above.")
print("💡 Tip: Some models might take time to download on first run.")
print("🔧 Install missing dependencies as needed: pip install torch transformers pillow")