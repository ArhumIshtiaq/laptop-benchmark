def extract_data_batch(texts, model, tokenizer, device):
    """Extracts data from a batch of resume texts."""
    all_data = []
    prompts = []

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
        # Corrected batch_chat call:  prompts is a positional argument
        responses = model.batch_chat(tokenizer, None, prompts, generation_config=dict(max_new_tokens=512, do_sample=False))


        for response in responses:
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
    else: #Fallback
        for prompt in prompts:
            response = model.chat(tokenizer, None, prompt, dict(max_new_tokens=512, do_sample=False))
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
