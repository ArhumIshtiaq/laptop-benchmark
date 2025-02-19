import streamlit as st
import pandas as pd
from PIL import Image
import io
import pdfminer
import docx2txt
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
import tempfile
import os

import torch
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

def load_internvl_model():
    """Loads the InternVL model and tokenizer (optimized for RTX GPU)."""
    path = "OpenGVLab/InternVL2_5-1B-MPO"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    try:
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().to(device)
    except Exception as e:
        st.error(f"Error loading model with quantization: {e}. "
                 "Falling back to loading without quantization.")
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True
        ).eval().to(device)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    return model, tokenizer, device


def extract_text_from_image(image_file, model, tokenizer, device):
    """Extracts text and description from an image (optimized)."""
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def build_transform(input_size):
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(image_file, input_size=448, max_num=12):
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        transform = build_transform(input_size=input_size)
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    pixel_values = load_image(image_file, max_num=12).to(torch.float16).to(device)
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    question = '<image>\nPlease describe the image in detail and extract all text in the image'
    response = model.chat(tokenizer, pixel_values, question, generation_config) #Revert to chat for images
    return response

def extract_data_batch(texts, model, tokenizer, device):
    """Extracts data from a batch of resume texts. Uses text-only batch processing."""
    all_data = []
    prompts = []
    data_templates = [] # Store data templates for each text

    for text in texts:
        data_template = {
            'Name': 'Not Mentioned',
            'Contact number': 'Not Mentioned',
            'Email address': 'Not Mentioned',
            'Gender': 'Not Mentioned',
            'Date of birth': 'Not Mentioned',
            'CNIC number': 'Not Mentioned',
            'Do you have any disability?': 'Not Mentioned',
            'Select your city': 'Not Mentioned',
            'Marital status': 'Not Mentioned',
            'Education': 'Not Mentioned',
            'Employment status': 'Not Mentioned',
            'Years of experience': '0',
            'If employed, share company name?': 'Not Mentioned',
            'Type of job preferred': 'Not Mentioned'
        }
        data_templates.append(data_template) #Keep track of the template

        prompt = f"""
Extract the following information from the resume text below.

Resume Text:
{text}

Instructions:

*   **Years of experience:**  Aggregate the total years of experience from ALL mentioned jobs, roles, or periods of employment.  Calculate durations from date ranges (e.g., 2018-2020 is 2 years). If no experience is explicitly mentioned, output 0.
*   **If employed, share company name?:**  Output the *CURRENT* employer, or the *MOST RECENT* employer if the person is currently unemployed. Use contextual clues like "currently working at," "present employer," verb tenses, and dates. If unemployed, output "Unemployed".
*   For all other fields, if a piece of information is not found, output 'Not Mentioned'.

Output in the following format, each on a new line:
Name: <name>
Contact number: <number>
Email address: <email>
Gender: <gender>
Date of birth: <dob>
CNIC number: <cnic>
Do you have any disability?: <disability>
Select your city: <city>
Marital status: <marital_status>
Education: <education>
Employment status: <employment_status>
Years of experience: <years_of_experience>
If employed, share company name?: <company_name>
Type of job preferred: <job_type>
"""
        prompts.append(prompt)

    if hasattr(model, 'batch_chat'):
        # Text-only batch processing.  Pass None for pixel_values and image_counts.
        responses = model.batch_chat(tokenizer, None, questions=prompts, generation_config=dict(max_new_tokens=512, do_sample=False))

        for response, data_template in zip(responses, data_templates):
            data = data_template.copy() # Use the correct data_template
            for line in response.strip().split("\n"):
                try:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    key_mapping = {
                        'Name': 'Name',
                        'Contact number': 'Contact number',
                        'Email address': 'Email address',
                        'Gender': 'Gender',
                        'Date of birth': 'Date of birth',
                        'CNIC number': 'CNIC number',
                        'Do you have any disability?': 'Do you have any disability?',
                        'Select your city': 'Select your city',
                        'Marital status': 'Marital status',
                        'Education': 'Education',
                        'Employment status': 'Employment status',
                        'Years of experience': 'Years of experience',
                        'If employed, share company name?': 'If employed, share company name?',
                        'Type of job preferred': 'Type of job preferred'
                    }
                    if key in key_mapping:
                        data[key_mapping[key]] = value
                except ValueError:
                    pass
            all_data.append(data)
    else: #Fallback to chat
        for prompt in prompts:
            response = model.chat(tokenizer, None, prompt, dict(max_new_tokens=512, do_sample=False)) #Use text-only chat
            data = data_template.copy()
            for line in response.strip().split("\n"):
                try:
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()
                    key_mapping = {
                        'Name': 'Name',
                        'Contact number': 'Contact number',
                        'Email address': 'Email address',
                        'Gender': 'Gender',
                        'Date of birth': 'Date of birth',
                        'CNIC number': 'CNIC number',
                        'Do you have any disability?': 'Do you have any disability?',
                        'Select your city': 'Select your city',
                        'Marital status': 'Marital status',
                        'Education': 'Education',
                        'Employment status': 'Employment status',
                        'Years of experience': 'Years of experience',
                        'If employed, share company name?': 'If employed, share company name?',
                        'Type of job preferred': 'Type of job preferred'
                    }
                    if key in key_mapping:
                        data[key_mapping[key]] = value
                except ValueError:
                    pass
            all_data.append(data)

    return all_data


def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)

    for page in PDFPage.get_pages(pdf_file, caching=True, check_extractable=True):
        page_interpreter.process_page(page)

    text = fake_file_handle.getvalue()
    converter.close()
    fake_file_handle.close()
    return text

def extract_text_from_docx(docx_file):
    """Extracts text from a DOCX file."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmpfile:
        tmpfile.write(docx_file.read())
        temp_filename = tmpfile.name
    try:
        text = docx2txt.process(temp_filename)
    finally:
        os.remove(temp_filename)
    return text

def extract_text_from_txt(txt_file):
    """Extract text from a plain text file."""
    return txt_file.read().decode("utf-8", errors="ignore")

def main():
    st.title("Resume Data Extractor")

    model, tokenizer, device = load_internvl_model()

    uploaded_files = st.file_uploader(
        "Upload resumes (PDF, DOCX, TXT, JPG, PNG)",
        type=["pdf", "docx", "txt", "jpg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Extract Data"):
            all_texts = []
            all_filenames = []

            with st.spinner('Loading files...'):
                for uploaded_file in uploaded_files:
                    file_type = uploaded_file.type
                    try:
                        if file_type == "application/pdf":
                            text = extract_text_from_pdf(uploaded_file)
                        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                            text = extract_text_from_docx(uploaded_file)
                        elif file_type == "text/plain":
                            text = extract_text_from_txt(uploaded_file)
                        elif file_type.startswith("image"):
                            text = extract_text_from_image(uploaded_file, model, tokenizer, device)
                        else:
                            st.error(f"Unsupported file type: {file_type}")
                            continue

                        all_texts.append(text)
                        all_filenames.append(uploaded_file.name)

                    except Exception as e:
                        st.error(f"Error loading {uploaded_file.name}: {e}")
                        continue

            if all_texts:
                with st.spinner('Processing resumes...'):
                    all_data = extract_data_batch(all_texts, model, tokenizer, device)
                    df = pd.DataFrame(all_data)
                    st.dataframe(df)
            else:
                st.warning("No resumes loaded successfully.")

if __name__ == "__main__":
    main()
