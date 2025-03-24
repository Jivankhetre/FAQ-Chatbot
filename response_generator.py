from vertexai.generative_models import GenerativeModel

def generate_response(prompt, context, gcs_uri, system_instructions, safety_settings, generation_config):
    model = GenerativeModel(
        model_name="gemini-1.5-flash-001",
        safety_settings=safety_settings,
        generation_config=generation_config,
        system_instruction=system_instructions
    )
    response = model.generate_content([prompt]).text
    return f"{response}\n* GCS URI: {gcs_uri}"