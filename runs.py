# import pandas as pd
# from langchain.chains import RetrievalQA
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS  # or Chroma, etc.
# from langchain.llms import HuggingFaceHub  # or OpenAI, etc.
# from langchain.document_loaders import TextLoader  # if needed
# from newcrag import MooligAI
# from langchain.prompts import PromptTemplate

# df = pd.read_csv("ayur_eval.csv")
# df2 = pd.read_csv("dataset2.csv")
# mooligai = MooligAI()


# retrieval_prompt_template = PromptTemplate(
#     input_variables=["combined_context", "question"],
#     template="""
# You are Moolig.AI, an expert in Ayurveda, the ancient Indian system of natural healing.
# Provide an **accurate, concise, and holistic** answer based on Ayurvedic knowledge.

# ### Context:
# {combined_context}

# ### User Question:
# {question}

# Response Guidelines:
# - Concise and straight to the point reponses
# - Speak like a friendly guide
# - Base answers on Ayurvedic principles
# - Provide direct and actionable advice
# - Reference classical sources when possible
# - Promote safe, traditional remedies
# """
# )

# x = 0
# predictions = []
# knownq = df2["Query"].tolist()
# for q in df["Query"]:
#     match = df2[df2['Query'] == q]
#     if not match.empty:
#         result = match.iloc[0]['Answer']
#         predictions.append(result)
#         x+=1
#         print(x)
#         continue
#     try:
#         pinecone_context = mooligai.get_context(q)
#         combined_context = f"{pinecone_context}"
#         prompt_text = retrieval_prompt_template.format(
#             combined_context=combined_context,
#             question=q
#         )
#         result = mooligai.generate_response(prompt_text)
#         print(result)
#     except Exception as e:
#         result = f"Error: {str(e)}"

#     predictions.append(result)

# df["Prediction"] = predictions
# df.to_csv("groundtruth_with_predictions.csv", index=False)


###### DF CHECK ######

# import pandas as pd
# fp = r"E:\COLLEGE\SEM 6\CIP\AyurRagBot\errors.csv"
# df = pd.read_csv(fp)

# print(df.columns)
# print(df.head())
# print(df.shape)

# x = 0
# for _, row in df.iterrows():
#     if row["score"] == 0:
#         x += 1
#         print(f"\nQuestion {x}:")
#         print(f"Q: {row["question"]}", f"GT: {row["ground_truth"]}", f"P: {row["prediction"]}", sep="\n")
#         print()

# print(x)

# errors = df[df["score"] == 0].copy()
# errors.to_csv("errors.csv", columns=["question", "ground_truth", "prediction"], index=False)



##### ERROR UPDATE #####

# import pandas as pd

# eval_df = pd.read_csv("evaluation_results.csv")
# error_df = pd.read_csv("error_results.csv")

# error_score_map = dict(zip(error_df["question"], error_df["score"]))

# mask = eval_df["question"].isin(error_score_map)
# eval_df.loc[mask, "score"] = eval_df.loc[mask, "question"].map(error_score_map)

# eval_df.to_csv("evaluation_results_updated.csv", index=False)

# total_score = eval_df["score"].sum()
# print(total_score)

# print(eval_df.head())



##### EVAL SCORES #####

import pandas as pd

eval_df = pd.read_csv("evaluation_results_updated.csv")

n = len(eval_df)                     # total examples
correct = eval_df["score"].sum()           # number of correct predictions
accuracy = correct / n                     # fraction correct

print(f"Total examples: {n}")
print(f"Correct      : {correct}")
print(f"Accuracy     : {accuracy:.4%}")
