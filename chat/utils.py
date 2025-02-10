# import openai
# import json

# def parse_nlu_input(system_prompt: str, user_message: str, schema_columns: list) -> dict:
#     """
#     Use GPT to parse `user_message` (in plain English) into a structured set of updates:
#       e.g. { "target_column": "status", "entity_column": "account_id", "time_frame": "1 month", ... }
    
#     `schema_columns` can be used to help GPT map user synonyms to real columns.
#     The system_prompt instructs GPT to ONLY produce valid JSON with possible keys:
#       - "target_column"
#       - "entity_column"
#       - "time_frame"
#       - "predictive_question"
#       - (You can add "time_column" or "time_frequency" as well)
#     """
#     # Build a “system” prompt that instructs GPT
#     prompt = f"""
#     System instructions:
#     {system_prompt}

#     The dataset columns are: {schema_columns}.
#     The user says: {user_message}

#     Please output only valid JSON with any fields that the user might be specifying.
#     Possible keys:
#     - "target_column"
#     - "entity_column"
#     - "time_frame"
#     - "predictive_question"
#     - "time_column"
#     - "time_frequency"

#     If the user tries to rename a column or says something like "entity will be 'account'", 
#     see if 'account' is close to any actual columns. If there's no match, do your best or leave it blank.
#     """

#     # response = openai.ChatCompletion.create(
#     #     model="gpt-4o-mini",  # or your desired model
#     #     messages=[{"role": "system", "content": prompt}],
#     #     temperature=0
#     # )


#     import openai

#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": "System context"},
#             {"role": "user", "content": "User question..."}
#         ]
#     )
#     # answer = response["choices"][0]["message"]["content"]


#     content = response["choices"][0]["message"]["content"]
#     try:
#         parsed = json.loads(content)
#         return parsed
#     except:
#         return {}



import openai
import json

# def parse_nlu_input(system_prompt: str, user_message: str, schema_columns: list) -> dict:
#     """
#     Use GPT to parse user_message (in plain English) into a structured set of updates:
#       e.g. { "target_column": "status", "entity_column": "account_id", "time_frame": "1 month", ... }
    
#     `schema_columns` can be used to help GPT guess the correct column names if user uses synonyms.
#     The system_prompt instructs GPT to ONLY produce valid JSON with the possible keys:
#       - "target_column"
#       - "entity_column"
#       - "time_frame"
#       - "predictive_question"
#       - "time_column"
#       - "time_frequency"
#     """
#     # Build a single prompt that includes system instructions and user context
#     # This is what GPT will see.
#     prompt = f"""
#     System instructions:
#     {system_prompt}

#     The dataset columns are: {schema_columns}.
#     The user says: {user_message}

#     Please output ONLY valid JSON (no extra text) with any of these possible keys:
#     - "target_column"
#     - "entity_column"
#     - "time_frame"
#     - "time_column"
#     - "time_frequency"
#     - "predictive_question"

#     If the user tries to rename a column or says something like "entity will be 'account'",
#     see if 'account' is close to any actual columns. If there's no close match, leave it blank.
#     """

#     try:
#         response = openai.ChatCompletion.create(
#             model="gpt-4o-mini",  # or "gpt-3.5-turbo" or whichever model you actually have
#             messages=[
#                 {"role": "system", "content": prompt}
#             ],
#             temperature=0
#         )
#         content = response["choices"][0]["message"]["content"].strip()
        
#         # Attempt to parse the content as JSON
#         parsed = json.loads(content)
#         return parsed

#     except Exception as e:
#         print("[ERROR] parse_nlu_input failed:", e)
#         return {}


import logging

logger = logging.getLogger(__name__)



def parse_nlu_input(system_prompt, user_message, schema_columns):
    """
    Enhanced NLU parsing with better validation
    """
    try:
        messages = [
            {"role": "system", "content": f"""
            You are an AI assistant helping to parse user input for predictive settings.
            Available columns: {', '.join(schema_columns)}
            
            Task: Analyze the user message and extract:
            1. Target column changes
            2. Entity column changes
            3. Time frame specifications
            4. Predictive questions
            5. Confirmation intents
            
            Return ONLY a JSON with these exact keys:
            {{
                "is_confirmation": boolean,
                "updates": {{
                    "target_column": string or null,
                    "entity_column": string or null,
                    "time_frame": string or null,
                    "predictive_question": string or null
                }}
            }}
            
            If the user mentions changing a target or predicting something specific, 
            include it in both target_column and predictive_question.
            """},
            {"role": "user", "content": user_message}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )
        
        # Parse the response
        parsed_response = json.loads(response.choices[0].message.content)
        
        # Validate the updates against schema
        updates = parsed_response.get('updates', {})
        validated_updates = {}
        
        # Handle target column update
        if updates.get('target_column'):
            target_col = updates['target_column'].lower()
            # Find exact or closest match in schema
            if target_col in [col.lower() for col in schema_columns]:
                validated_updates['target_column'] = next(
                    col for col in schema_columns if col.lower() == target_col
                )
                # Also update predictive question
                if not updates.get('predictive_question'):
                    validated_updates['predictive_question'] = f"How can we predict {validated_updates['target_column']}?"
        
        # Handle entity column update
        if updates.get('entity_column'):
            entity_col = updates['entity_column'].lower()
            if entity_col in [col.lower() for col in schema_columns]:
                validated_updates['entity_column'] = next(
                    col for col in schema_columns if col.lower() == entity_col
                )
        
        # Handle other updates
        if updates.get('time_frame'):
            validated_updates['time_frame'] = updates['time_frame']
        if updates.get('predictive_question'):
            validated_updates['predictive_question'] = updates['predictive_question']
        
        return {
            'is_confirmation': parsed_response.get('is_confirmation', False),
            'updates': validated_updates
        }
        
    except Exception as e:
        logger.error(f"Error in parse_nlu_input: {str(e)}")
        return {'is_confirmation': False, 'updates': {}}


import openai
import json

def classify_ml_type(user_question: str) -> str:
    """
    Uses GPT to decide if user_question implies classification or regression.
    Returns either "classification" or "regression" (or "unknown").
    """
    system_prompt = """
    You are a system that classifies the user’s predictive question:
    - If the question is about predicting a yes/no outcome, or churn, or categories, it's "classification".
    - If the question is about predicting a numeric or continuous value, it's "regression".
    Output valid JSON like: {"machine_learning_type": "classification"}
    """
    # Build the GPT request
    messages = [
      {"role": "system", "content": system_prompt},
      {"role": "user", "content": f"User question: {user_question}"}
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # or whichever model you have
            messages=messages,
            temperature=0
        )
        content = response["choices"][0]["message"]["content"].strip()
        data = json.loads(content)
        ml_type = data.get("machine_learning_type", "unknown")
        if ml_type in ["classification", "regression"]:
            return ml_type
        else:
            return "unknown"
    except:
        return "unknown"


def gpt_suggest_columns(schema_cols: list, user_chat_history: str) -> dict:
    """
    Ask GPT to pick the most likely target_column and entity_column
    given the list of schema_cols and any relevant user history.
    Return a dict:
      {
        "suggested_target_column": ...,
        "suggested_entity_column": ...
      }
    """
    system_prompt = f"""
    You are a system that receives a list of columns: {schema_cols}
    and a conversation history:\n{user_chat_history}\n
    Please figure out which column is most likely the 'target' for a prediction,
    and which column is the 'entity' or 'primary key' based on clues.
    Return valid JSON with keys "suggested_target_column" and "suggested_entity_column".
    """
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",   # Or whichever model you have
            messages=[{"role": "system", "content": system_prompt}],
            temperature=0
        )
        content = response["choices"][0]["message"]["content"].strip()
        return json.loads(content)
    except Exception as e:
        print("[ERROR] gpt_suggest_columns failed:", e)
        return {}
