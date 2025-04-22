import pandas as pd
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # or Chroma, etc.
from langchain.llms import HuggingFaceHub  # or OpenAI, etc.
from langchain.document_loaders import TextLoader  # if needed
from newcrag import MooligAI
from langchain.prompts import PromptTemplate

df = pd.read_csv("dataset.csv")
mooligai = MooligAI()


retrieval_prompt_template = PromptTemplate(
    input_variables=["combined_context", "question"],
    template="""
You are Moolig.AI, an expert in Ayurveda, the ancient Indian system of natural healing.
Provide an **accurate, concise, and holistic** answer based on Ayurvedic knowledge.

### Context:
{combined_context}

### User Question:
{question}

Response Guidelines:
- Concise and straight to the point reponses
- Speak like a friendly guide
- Base answers on Ayurvedic principles
- Provide direct and actionable advice
- Reference classical sources when possible
- Promote safe, traditional remedies
"""
)

x = 0
predictions = []
for q in df["Question"]:
    try:
        pinecone_context = mooligai.get_context(q)
        combined_context = f"{pinecone_context}"
        prompt_text = retrieval_prompt_template.format(
            combined_context=combined_context,
            question=q
        )
        answer = mooligai.generate_response(prompt_text)
    except Exception as e:
        result = f"Error: {str(e)}"

    if x<5:
        x += 1
        print(predictions)
    predictions.append(result)

df["Prediction"] = predictions
df.to_csv("groundtruth_with_predictions.csv", index=False)