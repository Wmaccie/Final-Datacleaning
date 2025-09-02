import os
import pandas as pd
from typing import Tuple, List, Dict, Optional, Any, Callable
import time
import traceback
import json
import io
import random
from datetime import datetime
import inspect

# (Imports... unchanged)
OpenAI: Optional[type] = None
APIError: type = Exception
RateLimitError: type = Exception
APIConnectionError: type = Exception
AuthenticationError: type = Exception
OPENAI_CLIENT_AVAILABLE: bool = False
openai_client: Optional[any] = None
_load_dotenv_func: Optional[callable] = None
DOTENV_AVAILABLE: bool = False
MANUAL_UTILS_FOR_AI_AVAILABLE: bool = False
CleaningSummary: Any = dict
_analyze_dataset_func: Optional[callable] = None
_remove_duplicates_func: Optional[callable] = None
_handle_missing_values_func: Optional[callable] = None
_standardize_values_func: Optional[callable] = None

try:
    from openai import OpenAI, APIError, RateLimitError, APIConnectionError, AuthenticationError
    OPENAI_CLIENT_AVAILABLE = True
except ImportError: print("CRITICAL WARNING (ai_utils.py): OpenAI library not found. Please install with: pip install openai")
try:
    from dotenv import load_dotenv
    _load_dotenv_func = load_dotenv
    DOTENV_AVAILABLE = True
except ImportError: print("WARNING (ai_utils.py): python-dotenv not found. API key should be set as an environment variable.")
try:
    from utils.cleaning_utils.manual_utils import (
        analyze_dataset, remove_duplicates, handle_missing_values, standardize_values, CleaningSummary
    )
    _analyze_dataset_func = analyze_dataset
    _remove_duplicates_func = remove_duplicates
    _handle_missing_values_func = handle_missing_values
    _standardize_values_func = standardize_values
    MANUAL_UTILS_FOR_AI_AVAILABLE = True
except ImportError as e_manual_final:
    print(f"CRITICAL WARNING (ai_utils.py): Could not import manual_utils.py: {e_manual_final}. AI will be limited.")
    def _ai_dummy_fallback_final(df: pd.DataFrame, **kwargs: Any) -> Tuple[pd.DataFrame, Dict, Optional[pd.DataFrame]]: return df.copy(), {"error": "Utility not available"}, None
    _remove_duplicates_func = lambda df, **kwargs: _ai_dummy_fallback_final(df, **kwargs)
    _handle_missing_values_func = lambda df, **kwargs: _ai_dummy_fallback_final(df, **kwargs)
    _standardize_values_func = lambda df, **kwargs: _ai_dummy_fallback_final(df, **kwargs)
    _analyze_dataset_func = lambda df: {"error": "analyze_dataset utility not available"}

# (API Key setup... unchanged)
if DOTENV_AVAILABLE and callable(load_dotenv):
    if load_dotenv(): print("INFO (ai_utils.py): .env file loaded.")
    else: print("INFO (ai_utils.py): .env file not found, relying on existing environment variables.")
OPENAI_API_KEY_VALUE = os.getenv("OPENAI_API_KEY")
if OPENAI_CLIENT_AVAILABLE and callable(OpenAI):
    if OPENAI_API_KEY_VALUE and OPENAI_API_KEY_VALUE.strip():
        try:
            openai_client = OpenAI(api_key=OPENAI_API_KEY_VALUE.strip(), timeout=30.0, max_retries=2)
            print("INFO (ai_utils.py): OpenAI client initialized successfully.")
        except Exception as e_init_client:
            print(f"ERROR (ai_utils.py): Could not initialize OpenAI client: {e_init_client}. AI is disabled.")
            openai_client = None; OPENAI_CLIENT_AVAILABLE = False
    else:
        print("CRITICAL WARNING (ai_utils.py): OPENAI_API_KEY not found or empty. AI is disabled.")
        OPENAI_CLIENT_AVAILABLE = False
else:
    if not OPENAI_CLIENT_AVAILABLE: print("CRITICAL WARNING (ai_utils.py): OpenAI library not imported correctly. AI is disabled.")
    OPENAI_CLIENT_AVAILABLE = False

AI_RESPONSE_TEMPLATES = {
    "error_api": "Uh oh! My hotline to the AI super-brain seems to be on a tea break ☕. Please try again in a moment!",
    "error_no_config": "My advanced cleaning circuits are offline (missing API key or tools). I can chat, but no deep cleaning now! 🙁",
    "error_parsing_ai": "Whoopsie! My translation circuits got a bit tangled. Could you rephrase that? 😵‍💫",
}

def get_df_context_for_prompt(df: Optional[pd.DataFrame], max_rows=5, max_cols=10, max_str_len=35) -> str:
    # (Function unchanged)
    if df is None or df.empty: return "The DataFrame is currently empty."
    sio = io.StringIO(); df.info(buf=sio, verbose=False, memory_usage=False); info_str = sio.getvalue()
    sample_df = df.head(max_rows).copy()
    if len(sample_df.columns) > max_cols: sample_df = sample_df.iloc[:, :max_cols]
    for col in sample_df.select_dtypes(include=['object', 'string']).columns: sample_df[col] = sample_df[col].str.slice(0, max_str_len)
    return f"Info:\n{info_str[:1000]}\n\nSample Data:\n{sample_df.to_markdown(index=False)}"

class DataCleaningBot:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.default_model = model_name
        self.client: Optional[OpenAI] = openai_client
        self.conversation_history: List[Dict[str, str]] = []

    def _choose_model(self, prompt: str) -> str:
        # (Function unchanged)
        if not self.client: return self.default_model
        try:
            decision_resp = self.client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": "You are a routing agent. Analyze request complexity. Simple chat is 'nano'. Standard cleaning is 'mini'. Complex analysis or code generation is 'full'. Return only 'nano', 'mini', or 'full'."},
                    {"role": "user", "content": f"User request: {prompt}"}
                ],
                max_completion_tokens=10,
            )
            decision = decision_resp.choices[0].message.content.lower()
            if "nano" in decision: return "gpt-5-nano"
            elif "full" in decision: return "gpt-5"
            else: return "gpt-5-mini"
        except Exception as e:
            print(f"WARNING (ai_utils): Could not choose model due to API error: {e}. Falling back to default.")
            return self.default_model

    # --- THIS IS THE UPDATED SECTION ---
    @staticmethod
    def _create_prompt(df_context_str: str, user_query: str, history: List[Dict[str,str]], is_small_dataset: bool) -> List[Dict[str, str]]:
        system_prompt = f"""
You are "Scrubbie", a witty and skilled data cleaning assistant. Your mission is to generate a JSON object that plans the next step.

INTENT options:
- "execute_clean": User wants a DIRECT cleaning action. ONLY choose this if `is_small_dataset` is true. You MUST provide `function_to_call` and `parameters`.
- "suggestion": User wants to clean, but `is_small_dataset` is false OR the request is ambiguous. Suggest a manual tool via `tool_to_open`.
- "analyze": General question about data quality.
- "query": Specific factual question about data content.
- "download": User wants the file.
- "chat": General conversation.

JSON response structure: {{"thought": "...", "intent": "...", "action_details": {{ ... }}, "user_explanation": "..."}}

--- EXAMPLES ---
Context: is_small_dataset = true, User Request: "get rid of duplicate rows"
JSON: {{"thought": "User wants to remove duplicates on a small dataset. I will execute it directly.", "intent": "execute_clean", "action_details": {{"function_to_call": "remove_duplicates", "parameters": {{"keep": "first"}} }}, "user_explanation": "Done! I've zapped those pesky duplicate rows for you."}}

Context: is_small_dataset = true, User Request: "fill missing values"
JSON: {{"thought": "User wants to fill missing values on a small dataset. The default 'mode' strategy is best.", "intent": "execute_clean", "action_details": {{"function_to_call": "handle_missing_values", "parameters": {{"strategy": "mode"}} }}, "user_explanation": "All set! I've filled in the empty spots using the most common value in each column."}}

Context: is_small_dataset = true, User Request: "make the 'name' column lowercase"
JSON: {{"thought": "User wants to standardize a text column to lowercase on a small dataset. I will execute this directly.", "intent": "execute_clean", "action_details": {{"function_to_call": "standardize_values", "parameters": {{"target_columns": ["name"], "text_to_lowercase": true}} }}, "user_explanation": "Consider it done. The 'name' column is now all lowercase."}}

Context: is_small_dataset = false, User Request: "get rid of duplicate rows"
JSON: {{"thought": "User wants to remove duplicates, but the dataset is large. I will suggest the manual tool for safety.", "intent": "suggestion", "action_details": {{"tool_to_open": "Duplicates", "suggested_parameters": {{"strategy": "first"}} }}, "user_explanation": "You bet! For this large dataset, I've set up the 'Duplicates' tool in the Manual Cleaning tab. Please review and confirm!"}}
---
CONTEXT:
- Current dataset is small: {is_small_dataset}
- Date: {datetime.now().strftime('%Y-%m-%d')}
"""
        user_content = f"DataFrame Context:\n{df_context_str}\n\nMy request is: '{user_query}'\n\nYour response MUST be a valid JSON object..."
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(history[-4:])
        messages.append({"role": "user", "content": user_content})
        return messages

    def respond(
            self,
            user_input: str,
            df: pd.DataFrame,
            ui_update_callback: Optional[Callable[[str], None]] = None
    ) -> Tuple[str, pd.DataFrame, Optional[Dict[str, Any]]]:
        # (Function logic remains unchanged)
        if not self.client or not OPENAI_CLIENT_AVAILABLE: return AI_RESPONSE_TEMPLATES["error_no_config"], df.copy(), None
        is_small_dataset = len(df) < 5000 if isinstance(df, pd.DataFrame) else True
        model_to_use = self._choose_model(user_input)
        if ui_update_callback: ui_update_callback(f"🧠 *Model: **{model_to_use}** ... Thinking... ▌")
        self.conversation_history.append({"role": "user", "content": user_input})
        df_context = get_df_context_for_prompt(df)
        messages_for_api = self._create_prompt(df_context, user_input, self.conversation_history, is_small_dataset)
        final_cleaned_df = df.copy()
        ai_explanation = AI_RESPONSE_TEMPLATES["error_api"]
        action_result: Dict[str, Any] = {}
        full_response_content = ""
        try:
            completion = self.client.chat.completions.create(
                model=model_to_use,
                messages=messages_for_api,
                response_format={"type": "json_object"},
                max_completion_tokens=2000
            )
            full_response_content = completion.choices[0].message.content or ""
            if ui_update_callback: ui_update_callback(f"🧠 *Model: **{model_to_use}***\n\n" + full_response_content)
            parsed_ai_plan = json.loads(full_response_content)
            ai_explanation = parsed_ai_plan.get("user_explanation", "I have a plan for you!")
            intent = parsed_ai_plan.get("intent")
            action_details = parsed_ai_plan.get("action_details", {}) or {}
            if intent == 'execute_clean':
                func_name = action_details.get("function_to_call")
                params = action_details.get("parameters", {}) or {}
                if func_name == "remove_duplicates" and "subset" in params: params["columns_to_consider"] = params.pop("subset")
                if func_name == "handle_missing_values" and "columns" in params: params["column_to_process"] = params.pop("columns")
                if func_name == "standardize_values" and "columns" in params: params["target_columns"] = params.pop("columns")
                function_map = {"remove_duplicates": _remove_duplicates_func, "handle_missing_values": _handle_missing_values_func, "standardize_values": _standardize_values_func}
                if func_name in function_map and callable(function_map[func_name]):
                    cleaning_func = function_map[func_name]
                    valid_params = set(inspect.signature(cleaning_func).parameters.keys())
                    filtered_params = {k: v for k, v in params.items() if k in valid_params}
                    df_cleaned, report_obj, trash_df = cleaning_func(final_cleaned_df, **filtered_params)
                    final_cleaned_df = df_cleaned
                    rows_affected = getattr(report_obj, 'rows_affected', 0)
                    ai_explanation += f"\n\n**Result:** I performed the action, and **{rows_affected} rows** were affected."
                    details_dict = getattr(report_obj, '__dict__', {})
                    action_result['log_action'] = {'action': f"AI Execute: {func_name}", 'details': details_dict}
                    if trash_df is not None and not trash_df.empty: action_result[f'trash_{func_name}'] = trash_df
                    action_result['ui_action'] = 'show_review_button'
            elif intent == 'suggestion':
                action_result['ui_action'] = 'prefill_tool'
                action_result['details'] = action_details
        except Exception as e:
            print(f"CRITICAL ERROR (ai_utils): General error in respond: {e}\n{traceback.format_exc()}")
            ai_explanation = "Oops! Something went wrong. Please try again."
            return ai_explanation, df.copy(), None
        final_explanation = f"🧠 *Model: **{model_to_use}***\n\n{ai_explanation}"
        self.conversation_history.append({"role": "assistant", "content": ai_explanation})
        return final_explanation, final_cleaned_df, action_result if action_result else None