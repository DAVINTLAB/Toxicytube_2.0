"""
LLM Classifier Backend
Contains all business logic for text classification using LiteLLM and DSPy
"""
import os
import gc
from typing import Optional, Dict, List, Any, Callable

# LiteLLM for unified LLM API access
import litellm

# DSPy for structured prompting
import dspy


# =============================================================================
# Available Models
# =============================================================================

LLM_MODELS = {
    # OpenAI Chat Completion Models
    'openai/gpt-5': {
        'name': 'GPT-5',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5-mini': {
        'name': 'GPT-5 Mini',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5-nano': {
        'name': 'GPT-5 Nano',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5-chat': {
        'name': 'GPT-5 Chat',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5-chat-latest': {
        'name': 'GPT-5 Chat (Latest)',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5-2025-08-07': {
        'name': 'GPT-5 (2025-08-07)',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5-mini-2025-08-07': {
        'name': 'GPT-5 Mini (2025-08-07)',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5-nano-2025-08-07': {
        'name': 'GPT-5 Nano (2025-08-07)',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5-pro': {
        'name': 'GPT-5 Pro',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5.2': {
        'name': 'GPT-5.2',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5.2-2025-12-11': {
        'name': 'GPT-5.2 (2025-12-11)',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5.2-chat-latest': {
        'name': 'GPT-5.2 Chat (Latest)',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5.2-pro': {
        'name': 'GPT-5.2 Pro',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5.2-pro-2025-12-11': {
        'name': 'GPT-5.2 Pro (2025-12-11)',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5.1': {
        'name': 'GPT-5.1',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5.1-codex': {
        'name': 'GPT-5.1 Codex',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5.1-codex-mini': {
        'name': 'GPT-5.1 Codex Mini',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-5.1-codex-max': {
        'name': 'GPT-5.1 Codex Max',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4.1': {
        'name': 'GPT-4.1',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4.1-mini': {
        'name': 'GPT-4.1 Mini',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4.1-nano': {
        'name': 'GPT-4.1 Nano',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/o4-mini': {
        'name': 'O4 Mini',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/o3-mini': {
        'name': 'O3 Mini',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/o3': {
        'name': 'O3',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/o1-mini': {
        'name': 'O1 Mini',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/o1-preview': {
        'name': 'O1 Preview',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4o-mini': {
        'name': 'GPT-4o Mini',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4o-mini-2024-07-18': {
        'name': 'GPT-4o Mini (2024-07-18)',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4o': {
        'name': 'GPT-4o',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4o-2024-08-06': {
        'name': 'GPT-4o (2024-08-06)',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4o-2024-05-13': {
        'name': 'GPT-4o (2024-05-13)',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4-turbo': {
        'name': 'GPT-4 Turbo',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4-turbo-preview': {
        'name': 'GPT-4 Turbo (Preview)',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4-0125-preview': {
        'name': 'GPT-4 0125 Preview',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4-1106-preview': {
        'name': 'GPT-4 1106 Preview',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-3.5-turbo-1106': {
        'name': 'GPT-3.5 Turbo 1106',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-3.5-turbo': {
        'name': 'GPT-3.5 Turbo',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-3.5-turbo-0301': {
        'name': 'GPT-3.5 Turbo 0301',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-3.5-turbo-0613': {
        'name': 'GPT-3.5 Turbo 0613',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-3.5-turbo-16k': {
        'name': 'GPT-3.5 Turbo 16k',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-3.5-turbo-16k-0613': {
        'name': 'GPT-3.5 Turbo 16k 0613',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4': {
        'name': 'GPT-4',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4-0314': {
        'name': 'GPT-4 0314',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4-0613': {
        'name': 'GPT-4 0613',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4-32k': {
        'name': 'GPT-4 32k',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4-32k-0314': {
        'name': 'GPT-4 32k 0314',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
    'openai/gpt-4-32k-0613': {
        'name': 'GPT-4 32k 0613',
        'provider': 'OpenAI',
        'api_key_env': 'OPENAI_API_KEY',
    },
}


def get_available_models() -> Dict[str, Dict]:
    """
    Returns the available LLM models.
    
    Returns:
        dict: Dictionary with model information
    """
    return LLM_MODELS


def get_providers() -> List[str]:
    """
    Returns unique list of providers.
    
    Returns:
        list: List of provider names
    """
    providers = set()
    for model_info in LLM_MODELS.values():
        providers.add(model_info['provider'])
    return sorted(list(providers))


def get_models_by_provider(provider: str) -> Dict[str, Dict]:
    """
    Returns models filtered by provider.
    
    Args:
        provider: Provider name (e.g., 'OpenAI', 'Anthropic')
        
    Returns:
        dict: Dictionary with model information for that provider
    """
    return {
        model_id: model_info
        for model_id, model_info in LLM_MODELS.items()
        if model_info['provider'] == provider
    }


# =============================================================================
# API Key Management
# =============================================================================

def set_api_key(api_key: str, provider: str) -> None:
    """
    Sets the API key for a specific provider.
    
    Args:
        api_key: The API key string
        provider: Provider name
    """
    env_vars = {
        'OpenAI': 'OPENAI_API_KEY',
        'Anthropic': 'ANTHROPIC_API_KEY',
        'Google': 'GEMINI_API_KEY',
        'Groq': 'GROQ_API_KEY',
        'Mistral AI': 'MISTRAL_API_KEY',
        'DeepSeek': 'DEEPSEEK_API_KEY',
    }

    if provider in env_vars:
        os.environ[env_vars[provider]] = api_key


def validate_api_key(model_id: str, api_key: str) -> tuple[bool, str]:
    """
    Validates an API key by making a simple test request.
    
    Args:
        model_id: The model ID to test
        api_key: The API key to validate
        
    Returns:
        tuple: (is_valid, message)
    """
    try:
        # Set the API key for the provider
        if model_id in LLM_MODELS:
            provider = LLM_MODELS[model_id]['provider']
            set_api_key(api_key, provider)

        # Make a simple test request
        response = litellm.completion(
            model=model_id,
            messages=[{"role": "user", "content": "Say 'OK' if you can read this."}],
            max_tokens=10,
            api_key=api_key
        )

        return True, "✅ API key validated successfully!"

    except litellm.AuthenticationError:
        return False, "❌ Invalid API key. Please check your credentials."
    except litellm.RateLimitError:
        return True, "⚠️ API key is valid but rate limited. Please wait and try again."
    except Exception as e:
        return False, f"❌ Error validating API key: {str(e)}"


# =============================================================================
# DSPy Configuration
# =============================================================================

def configure_dspy(model_id: str, api_key: str) -> tuple[Optional[object], Optional[str]]:
    """
    Configures DSPy with the specified model.
    Returns the LM object to be used in context, or error message.
    
    Args:
        model_id: The LiteLLM model ID
        api_key: The API key
        
    Returns:
        tuple: (lm_object, error_message) - if success, error_message is None
    """
    try:
        # Set API key
        if model_id in LLM_MODELS:
            provider = LLM_MODELS[model_id]['provider']
            set_api_key(api_key, provider)

        # Create LM instance
        lm = dspy.LM(model=model_id, api_key=api_key)

        return lm, None

    except Exception as e:
        return None, f"Error configuring DSPy: {str(e)}"


# =============================================================================
# Test Model Connection
# =============================================================================

def test_model_connection(model_id: str, api_key: str, test_message: str) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Tests the model connection by sending a message and receiving a response.
    Returns the response text, the model identifier reported in the response JSON (if present),
    and an optional error message.
    
    Args:
        model_id: The LiteLLM model ID
        api_key: The API key
        test_message: The message to send
        
    Returns:
        tuple: (response_text, model_from_response, error_message)
    """
    try:
        # Set API key for the provider
        if model_id in LLM_MODELS:
            provider = LLM_MODELS[model_id]['provider']
            set_api_key(api_key, provider)

        # Make the request using LiteLLM
        response = litellm.completion(
            model=model_id,
            messages=[{"role": "user", "content": test_message}],
            api_key=api_key
        )

        # Extract response text
        # Support both attribute-style and dict-style responses
        try:
            response_text = response.choices[0].message.content
        except Exception:
            # Fallbacks in case structure differs
            try:
                response_text = response['choices'][0]['message']['content']
            except Exception:
                response_text = str(response)

        # Try to extract the model identifier from the response JSON/object
        model_from_response = None
        try:
            model_from_response = getattr(response, 'model', None)
        except Exception:
            model_from_response = None

        if not model_from_response:
            try:
                model_from_response = response.get('model') if isinstance(response, dict) else None
            except Exception:
                model_from_response = None

        return response_text, model_from_response, None

    except litellm.AuthenticationError:
        return None, None, "Authentication failed. Please check your API key."
    except litellm.RateLimitError:
        return None, None, "Rate limit exceeded. Please wait and try again."
    except litellm.APIConnectionError:
        return None, None, "Connection error. Please check your internet connection."
    except Exception as e:
        return None, None, f"Error: {str(e)}"


# =============================================================================
# DSPy Signature for Text Classification
# =============================================================================

class TextClassifier(dspy.Signature):
    """Classify text according to the provided instructions."""

    classification_instructions: str = dspy.InputField(desc="Classification instructions and label definitions")
    text: str = dspy.InputField(desc="Text to be classified")
    classification: str = dspy.OutputField(desc="The classification label for the text")
    confidence: str = dspy.OutputField(desc="Confidence level: high, medium, or low")
    reasoning: str = dspy.OutputField(desc="Brief reasoning for the classification")


class TextClassifierWithLabels(dspy.Signature):
    """Classify text according to the provided instructions. You must choose one of the provided labels."""

    classification_instructions: str = dspy.InputField(desc="Classification instructions")
    labels: str = dspy.InputField(desc="Available classification labels (comma-separated)")
    text: str = dspy.InputField(desc="Text to be classified")
    classification: str = dspy.OutputField(desc="The classification label (must be one of the provided labels)")
    confidence: str = dspy.OutputField(desc="Confidence level: high, medium, or low")
    reasoning: str = dspy.OutputField(desc="Brief reasoning for the classification")


# =============================================================================
# Classification Functions
# =============================================================================

def classify_single_text_llm(
    text: str,
    instructions: str,
    labels: Optional[str],
    model_id: str,
    api_key: str
) -> tuple[Optional[Dict], Optional[str], Optional[str]]:
    """
    Classifies a single text using the LLM.
    
    Args:
        text: Text to classify
        instructions: Classification instructions
        labels: Optional comma-separated labels
        model_id: The LiteLLM model ID
        api_key: The API key
        
    Returns:
        tuple: (result_dict, error_message)
    """
    try:
        # Configure DSPy
        lm, error_msg = configure_dspy(model_id, api_key)
        if not lm:
            return None, error_msg

        with dspy.context(lm=lm):
            # Create predictor based on whether labels are provided
            if labels and labels.strip():
                predictor = dspy.Predict(TextClassifierWithLabels)
                result = predictor(
                    classification_instructions=instructions,
                    labels=labels,
                    text=text
                )
            else:
                predictor = dspy.Predict(TextClassifier)
                result = predictor(
                    classification_instructions=instructions,
                    text=text
                )

            # Try to capture a raw response representation (best-effort)
            raw_response = None
            try:
                raw_response = getattr(result, '_raw', None)
            except Exception:
                raw_response = None

            if not raw_response:
                try:
                    # Fallback to dict or string representation
                    raw_response = dict(result.__dict__)
                except Exception:
                    try:
                        raw_response = str(result)
                    except Exception:
                        raw_response = None

            return {
                'classification': result.classification,
                'confidence': result.confidence,
                'reasoning': result.reasoning
            }, raw_response, None

    except Exception as e:
        return None, None, str(e)


def classify_texts_llm(
    texts: List[str],
    instructions: str,
    labels: Optional[str],
    model_id: str,
    api_key: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> tuple[Optional[List[Dict]], Optional[str]]:
    """
    Classifies a list of texts using the LLM.
    
    Args:
        texts: List of texts to classify
        instructions: Classification instructions
        labels: Optional comma-separated labels
        model_id: The LiteLLM model ID
        api_key: The API key
        progress_callback: Optional callback function (current, total)
        
    Returns:
        tuple: (results_list, error_message)
    """
    try:
        # Configure DSPy
        lm, error_msg = configure_dspy(model_id, api_key)
        if not lm:
            return None, error_msg

        with dspy.context(lm=lm):
            # Create predictor based on whether labels are provided
            if labels and labels.strip():
                predictor = dspy.Predict(TextClassifierWithLabels)
            else:
                predictor = dspy.Predict(TextClassifier)

            all_results = []
            total_texts = len(texts)

            for i, text in enumerate(texts):
                try:
                    if labels and labels.strip():
                        result = predictor(
                            classification_instructions=instructions,
                            labels=labels,
                            text=text
                        )
                    else:
                        result = predictor(
                            classification_instructions=instructions,
                            text=text
                        )

                    all_results.append({
                        'classification': result.classification,
                        'confidence': result.confidence,
                        'reasoning': result.reasoning,
                        'error': None
                    })

                except Exception as e:
                    all_results.append({
                        'classification': 'ERROR',
                        'confidence': 'low',
                        'reasoning': f'Classification error: {str(e)}',
                        'error': str(e)
                    })

                # Update progress
                if progress_callback:
                    progress_callback(i + 1, total_texts)

        return all_results, None

    except Exception as e:
        return None, str(e)


# =============================================================================
# Results Processing
# =============================================================================

def create_results_dataframe(
    original_df,
    text_column: str,
    classification_results: List[Dict]
):
    """
    Creates DataFrame with classification results.
    
    Args:
        original_df: Original DataFrame
        text_column: Name of the text column
        classification_results: List of classification result dicts
        
    Returns:
        pd.DataFrame: DataFrame with results
    """
    import pandas as pd

    results_df = original_df.copy()

    # Add classification columns
    results_df['llm_classification'] = [r['classification'] for r in classification_results]
    results_df['llm_confidence'] = [r['confidence'] for r in classification_results]
    results_df['llm_reasoning'] = [r['reasoning'] for r in classification_results]
    results_df['llm_error'] = [r.get('error') for r in classification_results]

    return results_df


def save_classification_results(results_df, output_path: str, output_format: str = 'csv') -> Dict:
    """
    Saves classification results to a file.
    
    Args:
        results_df: DataFrame with results
        output_path: Full file path
        output_format: Output format ('csv', 'xlsx', 'json')
        
    Returns:
        dict: {'success': bool, 'path': str, 'error': str}
    """
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Save based on format
        if output_format == 'csv':
            results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif output_format == 'xlsx':
            results_df.to_excel(output_path, index=False, engine='openpyxl')
        elif output_format == 'json':
            results_df.to_json(output_path, orient='records', force_ascii=False, indent=2)
        else:
            return {
                'success': False,
                'error': f'Unsupported format: {output_format}'
            }

        return {
            'success': True,
            'path': output_path
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }


# =============================================================================
# Cleanup
# =============================================================================

def cleanup():
    """
    Cleans up resources.
    """
    gc.collect()
