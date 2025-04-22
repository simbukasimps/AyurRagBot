import os
from typing import List, Dict, Any
import pandas as pd
from langchain.chat_models import init_chat_model
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

os.environ["GROQ_API_KEY"] = "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"

class RAGEvaluator:
    def __init__(self, model_name: str = "deepseek-ai/deepseek-v2-chat"):
        self.llm = init_chat_model("deepseek-r1-distill-llama-70b", model_provider="groq")

        self.eval_prompt = ChatPromptTemplate.from_template(
            """You are an expert evaluator for Ayurvedic question answering systems.
            
            I will provide you with:
            1. A question about Ayurveda
            2. The ground truth answer from authoritative sources
            3. The predicted answer from a RAG system
            
            Your task is to evaluate if the predicted answer is better than the ground turth and accurate.
            
            Question: {question}
            
            Ground Truth Answer: {ground_truth}
            
            Predicted Answer: {prediction}
            
            First, analyze both answers carefully.
            Then determine if the predicted answer contains the same idea as the ground truth answer.
            Prefer the Prediction Answer if it is accurate and better response than the ground truth.

            Finally, output ONLY a 1 or 0:
            - Output 1 if the prediction is better or it accurately captures the essential information in the ground truth
            - Output 0 if the prediction is inaccurate, contains incorrect information, or misses key details
            
            Your output should be ONLY the number 1 or 0, with no additional text."""
        )
        
        self.eval_chain = self.eval_prompt | self.llm | StrOutputParser()
    
    def evaluate_single(self, question: str, ground_truth: str, prediction: str) -> int:
        result = self.eval_chain.invoke({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction
        })
        
        try:
            xx = int(result[-1].strip())
            print(xx)
            return xx
        except ValueError:
            print(f"Warning: Unexpected evaluation result - {result}")
            xx = 0
            print(xx)
            return xx 
    
    def evaluate_dataset(self, dataset: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, Any]:
        """Evaluate entire dataset and generate metrics"""
        if len(dataset) != len(predictions):
            raise ValueError("Dataset and predictions must have the same length")
        
        results = []
        for i, (entry, prediction) in enumerate(zip(dataset, predictions)):
            question = entry["Query"]
            ground_truth = entry["Answer"]
            
            print(f"Evaluating question {i+1}/{len(dataset)}")
            score = self.evaluate_single(question, ground_truth, prediction)
            
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "score": score
            })
        
        df_results = pd.DataFrame(results)
        accuracy = df_results["score"].mean()
        
        return {
            "detailed_results": results,
            "accuracy": accuracy,
            "total_correct": df_results["score"].sum(),
            "total_questions": len(dataset)
        }

def load_dataset(file_path: str) -> List[Dict[str, str]]:
    df = pd.read_csv(file_path)
    return df.to_dict('records')

def get_rag_predictions(dataset: List[Dict[str, str]]) -> List[str]:
    """
    This function should be replaced with calls to your actual RAG system
    It should return a list of predictions corresponding to each question in the dataset
    """
    
    # predictions = []
    # for item in dataset:
    #     for q, a in item.items():
    #         # get RAG pred for q
    
    import pandas as pd
    fp = r"E:\COLLEGE\SEM 6\CIP\AyurRagBot\errors.csv"
    df = pd.read_csv(fp)
    return df["Prediction"].tolist()

if __name__ == "__main__":
    dataset = load_dataset(r"E:\COLLEGE\SEM 6\CIP\AyurRagBot\errors.csv")
    predictions = get_rag_predictions(dataset)

    print(f"Loaded {len(dataset)} questions from dataset")
    print(f"Generated {len(predictions)} predictions")

    evaluator = RAGEvaluator()
    evaluation_results = evaluator.evaluate_dataset(dataset, predictions)
    
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {evaluation_results['accuracy']:.2%}")
    print(f"Correct: {evaluation_results['total_correct']}/{evaluation_results['total_questions']}")
    
    pd.DataFrame(evaluation_results['detailed_results']).to_csv("error_results.csv", index=False)
    print("Detailed results saved to evaluation_results.csv")