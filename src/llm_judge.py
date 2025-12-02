import os
import json
from typing import List, Tuple
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# format llm judge response
class LLMJudgeScores(BaseModel):
    factuality: int = Field(..., description="score from 1 to 5 for factual consistency with the source document")
    coherence: int = Field(..., description="Score from 1 to 5 for logical flow and structural organization of the response")


def invoke_llm_judge(predictions: List[str], references: List[str], sources: List[str]) -> Tuple[List[int], List[int]]:
    factuality_scores = []
    coherence_scores = []

    SYSTEM_PROMPT = (
        "You are an expert evaluator. Your task is to score a generated answer based on two criteria: "
        "Factuality/Faithfulness and Response Coherence. You must only output a single JSON object. "
        "The scoring scale is 1 (Poor) to 5 (Excellent) for both metrics. "
        "Strictly adhere to the following JSON schema:\n"
        "{'factuality': int, 'coherence': int}"
    )

    # scoring
    for i, (prediction, reference, source) in enumerate(zip(predictions, references, sources)):
        print(f"-> scoring response {i+1}/{len(predictions)} with gpt-4o...")
        
        user_prompt = f"""
        **Factuality/Faithfulness:** Score the response (1-5) based on factual consistency with the provided source document.
        - 1: Fails to address the question or contains major factual errors not supported by the source.
        - 5: Entirely accurate and directly supported by the source document.
        
        **Response Coherence:** Score the response (1-5) based on its logical flow and structural organization.
        - 1: Disorganized, confusing structure, abrupt transitions.
        - 5: Perfectly logical flow, easy to read, and well-organized.
        ---
        **Source Document:**
        {source}

        **Ground Truth Answer (Reference):**
        {reference}

        **Generated RAG Response (Prediction):**
        {prediction}
        ---
        Your JSON output MUST ONLY contain the scores for 'factuality' and 'coherence'.
        """
        
        load_dotenv()
        openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        res = openai_client.chat.completions.create(
            model='gpt-5-mini',
            messages=[
                {'role': 'developer', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': user_prompt},
            ],
            response_format={'type': 'json_object'} # return json - https://platform.openai.com/docs/api-reference/chat/create#chat_create-response_format
        )
        json_str = res.choices[0].message.content
        scores_data = json.loads(json_str) # type: ignore 
        
        scores = LLMJudgeScores(**scores_data)
        factuality_scores.append(scores.factuality)
        coherence_scores.append(scores.coherence)

    return factuality_scores, coherence_scores
