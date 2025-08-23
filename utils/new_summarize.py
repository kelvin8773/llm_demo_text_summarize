from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_sentences=10):
    summary = summarizer(
        text,
        max_length=200,   # tokens, controls brevity
        min_length=25,
        do_sample=False
    )[0]['summary_text']
    # Optional: split into sentences and trim to target
    sentences = summary.split('. ')
    return '. '.join(sentences[:max_sentences]).strip()


# paraphraser = pipeline("text2text-generation", model="Vamsi/T5_Paraphrase_Paws")

# def humanize_summary(summary):
#     prompt = f"Make this sound conversational but still professional: {summary}"
#     return paraphraser(prompt, max_length=60, do_sample=False)[0]['generated_text']

# def enhance_summary(text, max_sentences=3):
#     summary = summarize_text(text, max_sentences)
#     return humanize_summary(summary)
