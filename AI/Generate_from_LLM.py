import torch
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import re

def initialize_llm(key):
    login(key)
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cuda",
        use_auth_token=True
    )

    generator = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        pad_token_id=tokenizer.eos_token_id
    )

    return generator

def generate_clinical_data_from_llm(prompt, generator):
    output = generator(
        prompt,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.1,
        do_sample=True
    )[0]["generated_text"]

    structured_output = extract_radiology_sections(output)
    return structured_output

def extract_radiology_sections(output_text):
    patterns = {
        "impression": r"\*\*\(1\) Impression\*\*\n+(.*?)(?=\n\*\*\(2\) Diagnosis\*\*)",
        "diagnosis": r"\*\*\(2\) Diagnosis\*\*\n+(.*?)(?=\n\*\*\(3\) Recommendations\*\*)",
        "recommendations": r"\*\*\(3\) Recommendations\*\*\n+(.*)"
    }

    extracted = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output_text, re.DOTALL)
        content = match.group(1).strip() if match else ""
        content = re.sub(r"\n{2,}", "\n", content)  # Remove extra line breaks
        extracted[key] = content

    return extracted

def format_summary(data):
    return (
        f"- Location: {data['anatomical_location']}.\n"
        f"- Enhancing component: {data['enhancing_percentage']:.2f}% ({data['enhancing_tumor_volume']:.2f} cc)\n"
        f"- Necrotic component: {data['necrosis_percentage']:.2f}% ({data['necrosis_volume']:.2f} cc)\n"
        f"- Edema: {data['edema_percentage']:.2f}% ({data['edema_volume']:.2f} cc)\n"
        f"- Tumor core (enhancing + necrotic): {data['tumor_core_volume']:.2f} cc\n"
        f"- Total lesion volume: {data['whole_tumor_volume']:.2f} cc ({data['whole_tumor_percentage']:.2f}% of total brain volume)"
    )

def prepare_prompt(data):
    formatted_summary = format_summary(data)
    
    prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id|>
        You are a professional radiologist assistant. Given structured brain tumor segmentation data of an MRI scan, generate a formal radiology report.
        
        Follow these rules:
        - Do NOT repeat or paraphrase the raw numerical data.
        - The **Impression** section should provide a clinical interpretation of the findings, such as potential impact on brain function, mass effect, or anatomical disruption, not the location that is already given.
        - The **Diagnosis** section should state the most probable pathological entity based on imaging features, using precise medical terminology for the grade and type. It should also emphasize that definitive diagnosis requires histopathological and molecular analysis.
        - Use clear section headers exactly as: **(1) Impression**, **(2) Diagnosis**, and **(3) Recommendations**.
        - Write in a formal, clinical tone appropriate for inclusion in a radiology report.
        - Use bullet points (circles not asterisks) only inside the Recommendations section for actionable steps or follow-up imaging.
        - Do NOT mention or assume patient age, gender, or clinical history.
        
        <|start_header_id|>user<|end_header_id|>
        {formatted_summary}
        
        Generate a clinical impression, likely diagnosis, and clinical recommendations without repeating the data above.
        <|start_header_id|>assistant<|end_header_id|>
        """
    return prompt