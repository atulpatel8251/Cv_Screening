import openai
import PyPDF2
import io
import json
import hashlib
import re
from datetime import datetime
import pandas as pd
import streamlit as st
from PIL import Image
import base64
import docx
import docx2txt
import logging
import os
from dateutil import parser
from dateutil.relativedelta import relativedelta
import tempfile
from spire.doc import Document
#from dotenv import load_dotenv
# load_dotenv()
#from langchain.prompts import PromptTemplate
logging.basicConfig(filename='app.log', level=logging.ERROR)

# Set Streamlit to use wide mode for the full width of the page
st.set_page_config(page_title="Resume Scanner", page_icon=":page_facing_up:",layout="wide")

# Load the logo image
logo_image = Image.open('assests/HD_Human_Resources_Banner.jpg')

# Optionally, you can resize only if the original size is too large
# For high-definition display, consider not resizing if the image is already suitable
resized_logo = logo_image.resize((1500, 300), Image.LANCZOS)  # Maintain quality during resizing

# Display the logo
st.image(resized_logo)  # Dynamically adjust width

# Function to add background image from a local file
def add_bg_from_local(image_file,opacity=0.7):
    with open(image_file, "rb") as image:
        encoded_image = base64.b64encode(image.read()).decode()

    # Inject custom CSS for background
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: linear-gradient(rgba(255, 255, 255, {opacity}), rgba(255, 255, 255, {opacity})),url("data:assests/logo.jfif;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image path
#add_bg_from_local('assests/OIP.jfif')  # Adjust path to your image file

# Add styled container for the title and description with a maximum width
st.markdown("""
    <div style="background-color: lightblue; padding: 20px; border-radius: 10px; text-align: center; 
                 max-width: 450px; margin: auto;">
        <p style="color: black; margin-left: 20px; margin-top: 20px; font-weight: bold;font-size: 40px;">CV Screening Portal</p>
        <p style="color: black; margin-left: 15px; margin-top: 20px; font-size: 25px;">AI based CV screening</p>
    </div>
""", unsafe_allow_html=True)

# Change background color using custom CSS
st.markdown(
    """
    <style>
    body {
               background-color: green;

    }
    </style>
    """,
    unsafe_allow_html=True
)

# Hide Streamlit's default footer and customize H1 style
hide_streamlit_style = """
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

with open('style.css') as f:
 st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True) 
# Example: Add your image handling or other logic here
images = ['6MarkQ']


openai.api_key = st.secrets["secret_section"]["OPENAI_API_KEY"]

# Function to extract text from PDF uploaded via Streamlit


def extract_text_from_doc(file_path):
    """
    Extract text from a .doc file using Spire.Doc
    
    Args:
        file_path: Path to the .doc file
    
    Returns:
        str: Extracted text from the file
    """
    # Create a Document object
    document = Document()
    
    # Load the Word document
    document.LoadFromFile(file_path)

    # Extract the text of the document
    document_text = document.GetText()
    document.Close()  # Close the document
    return document_text

def extract_text_from_uploaded_pdf(uploaded_file):
    """
    Extract text from an uploaded PDF, DOCX, or DOC file.

    Args:
        uploaded_file: A file-like object containing the document

    Returns:
        str: Extracted text from the file or an empty string if an error occurs
    """
    try:
        # Determine the file type
        file_type = uploaded_file.name.split('.')[-1].lower()

        if file_type == "pdf":
            # Extract text from PDF
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = "\n".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
            return text.strip()

        elif file_type == "docx":
            # Extract text from DOCX
            doc = docx2txt.process(uploaded_file)
            #print(doc)
            return doc.strip()

        elif file_type == "doc":
            # For .doc files, we need to save it temporarily and use Spire.Doc
            with tempfile.NamedTemporaryFile(delete=False, suffix='.doc') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            try:
                # Extract text using Spire.Doc
                text = extract_text_from_doc(tmp_file_path)
                return text
            finally:
                # Clean up the temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)

        else:
            # Show error message for unsupported file types
            st.error("This file type is not supported. Please upload only PDF, DOCX, or DOC files.", icon="🚫")
            return ""

    except Exception as e:
        st.error(f"Error reading file: {e}", icon="🚫")
        logging.error(f"Error reading file: {e}")
        return ""
    
def use_genai_to_extract_essential_criteria(jd_text):
    prompt = (
        "Extract and structure the following details from the job description: "
        "1. Education requirements (mandatory qualifications for the role) "
        "2. Required experience (minimum required) "
        "3. Mandatory skills (skills the candidate must have) "
        "4. Certifications: Separate into 'Essential Certifications' and 'Desired Certifications' based on language in the JD. "
        "   Certifications with words like 'mandatory', 'required', or 'essential' should be in 'Essential Certifications'. "
        "   Certifications with words like 'preferred', 'nice-to-have', 'Desirable Skill', 'Desirable Criteria' or 'optional' should be in 'Desired Certifications'."
        "5. Desired skills (additional skills for brownie points). "
        "The job description is as follows:\n\n"
        f"{jd_text}\n\n"
        "Please provide the response as a JSON object with keys: 'education', 'experience', "
        "'skills', 'essential_certifications', 'desired_certifications', 'desired_skills'."
    )
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            extracted_criteria = json.loads(content)
            return extracted_criteria
        except json.JSONDecodeError:
            st.error("Failed to parse JSON from AI response. Here's the raw response:")
            st.write(content)
            return {}
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return {}

def classify_criteria(criteria_json):
    essential_criteria = {
        "Education": criteria_json.get("education", "Not specified"),
        "Experience": criteria_json.get("experience", "Not specified"),
        "Skills": criteria_json.get("skills", "Not specified"),
        "Essential Certifications": criteria_json.get("essential_certifications", "Not specified"),
    }

    desired_skills = {
        "Desired Certifications": criteria_json.get("desired_certifications", "Not specified"),
        "Desired Skills": criteria_json.get("desired_skills", "Not specified")
    }

    return essential_criteria, desired_skills
# Function to use GenAI to extract criteria from job description
def use_genai_to_extract_criteria(jd_text):
    prompt = (
        "Extract and structure the following details from the job description: "
        "1. Education requirements "
        "2. Required experience "
        "3. Mandatory skills "
        "4. Certifications "
        "5. Desired skills (for brownie points). "
        "The job description is as follows:\n\n"
        f"{jd_text}\n\n"
        "Please provide the response as a JSON object. For example:\n"
        "{\"education\": \"Bachelor's Degree, Master's Degree\", "
        "\"experience\": \"5 years experience in data science\", "
        "\"skills\": \"Python, SQL, Machine Learning\", "
        "\"Certifications\": \"AWS Certified, PMP\", "
        "\"desired_skills\": \"Deep Learning, NLP\"}"
    )
    
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a helpful assistant."},
                      {"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0
        )
        
        content = response.choices[0].message.content.strip()
        
        try:
            return content
        except json.JSONDecodeError:
            st.error("Failed to parse JSON from AI response. Here's the raw response:")
            st.write(content)
            return ""
    except Exception as e:
        st.error(f"Error with OpenAI API: {e}")
        return ""
    
def calculate_skill_score(skill_scores):
    # Sum up all the skill scores
    total_score = sum(skill_scores.values())    
    return total_score

# Function to extract total years of experience using OpenAI's GPT model
@st.cache_data
def extract_experience_from_cv(cv_text):
    # Get the current year dynamically
    x = datetime.now()
    current_year = x.strftime("%m-%Y")
    
    #print("CV_Text : ",cv_text)
    # Enhanced prompt template specifically for handling overlapping dates
    prompt_template = f"""
    Please analyze this  {cv_text} carefully to calculate the total years of professional experience. Follow these steps:
    
    1. First, list out all date ranges found in chronological order:
       - Replace 'Current' or 'till date' with {current_year}
       - Include all years mentioned with positions Formats from following 
       - Format as YYYY-YYYY for each position
       - Format as DD-MM-YYYY - DD-MM-YYYY(For example:10-Jul-12 to 31-Jan-21)
       - Format as YYYY-MM-DD - YYYY-MM-DD(For example:12-Jul-10 to 21-Jan-31)
       - Format as YYYY-DD-MM - YYYY-DD-MM(For example:12-10-Jul to 21-31-Jan)
       - Format as MM-YYYY-MM-YYYY
       - Format as YYYY-MM-YYYY-MM
       - Format as YYYY-YY.
       - if in the cv_text 'start year' and 'present year' or 'end year' are not mentioned,then extract experience from 'cvtext' , if also not mentioned experience in cvtext then return 0.
    2. Then, merge overlapping periods :
       - Identify any overlapping years
       - Only count overlapping periods once
       - Create a timeline of non-overlapping periods

    3. Calculate total experience:
       - Sum up all non-overlapping periods.
       - Round to one decimal place Return in Sngle Value.
       -     
    """

    try:
        # Query GPT-4 API with enhanced system message
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert in analyzing cv_text and calculating professional experience 
                     Pay special attention to overlapping date ranges(If two or more projects within the same company 
                     overlap, merge them into a single continuous range.) and ensure no double-counting of experience.
                     Always show your work step by step."""
                },
                {"role": "user", "content": prompt_template}
            ],
            max_tokens=1000,
            temperature=0
        )
        
        # Extract GPT response
        gpt_response = response.choices[0].message.content.strip()
        #print("gpt_response:",gpt_response)
        # Handle "present" or "current year" in the response
        gpt_response = gpt_response.replace("present", str(current_year)).replace("current", str(current_year))
        #print("gpt_response:",gpt_response)
        # Extract experience and start year using improved regex
        experience_match = re.findall(r'(\d+(?:\.\d+)?)\s*years?', gpt_response, re.IGNORECASE)
        #print("experience_match:",experience_match)
        start_year_match = re.search(r'Start Year:\s*(\d{4})', gpt_response, re.IGNORECASE)
        
        # Extract and convert values
        if experience_match:
    # Choose the most relevant value by looking at the context or largest value
            total_experience = max(map(float, experience_match))
            #print("total_experience:", total_experience)
            total_experience = str(round(total_experience, 1))  # Round to one decimal place
            #print("total_experience2:", total_experience)
        else:
            total_experience = "Not found"

            
        start_year = start_year_match.group(1) if start_year_match else "Not found"
        
        # Debugging output
        # print("\nFull GPT Response:", gpt_response)
        # print("\nExtracted Total Experience:", total_experience)
        # print("Extracted Start Year:", start_year)
        
        return {
            "total_years": total_experience,
            "start_year": start_year,
            "gpt_response": gpt_response
        }
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return {
            "total_years": "Not found",
            "start_year": "Not found",
            "error": str(e)
        }

cv_cache = {}

def normalize_cv_text(cv_text):
    """ Normalize CV text for consistent hashing. """
    return ' '.join(cv_text.strip().split()).lower()

def generate_cv_hash(cv_text):
    """ Generate a consistent hash for the normalized CV text. """
    normalized_text = normalize_cv_text(cv_text)
    return hashlib.sha256(normalized_text.encode()).hexdigest()

def calculate_skill_score(skill_scores):
    """
    Calculate the total skill score from the skill stratification.
    Returns the sum of all skill scores.
    
    Args:
        skill_scores (dict): Dictionary of skills and their scores
    
    Returns:
        float: Total sum of skill scores (not averaged)
    """
    if not skill_scores:
        return 0
    
    # Calculate the total sum of scores
    total_score = sum(skill_scores.values())
    
    return total_score
def extract_candidate_name_from_cv(cv_text):
    """
    Extracts the candidate's name from the {cv_text} content.

    Args:
        cv_text (str): The {cv_text} content.

    Returns:
        str: Extracted candidate name or 'Unknown Candidate' if not found.
    """
    try:
        # Extract top few lines of the CV for name extraction
        # cv_header = "\n".join(cv_text.splitlines()[:50])  # Top 5 lines for context
        # print("cvheader:",cv_header)
        # Prompt for GPT
        prompt = (
            "Extract the candidate's full name from this cv_text. The name is likely at the top "
            "'Name:', 'Resume of:', 'Name of Staff','name of candidate','first name last name', or similar. Ignore job titles, contact details, and other information.\n\n"
            f"Name:\n{cv_text}"
        )

        # GPT call
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a name extraction specialist. Extract candidate name only from a {cv_text}."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=30,
            temperature=0
        )

        # Extracted name
        candidate_name = response.choices[0].message.content.strip()
        #print('candidate_name:',candidate_name)
        # Validate and return
        if candidate_name and len(candidate_name.split()) <= 4:
            return candidate_name
        return "Unknown Candidate"

    except Exception as e:
        logging.error(f"Error extracting candidate name: {e}")
        return "Unknown Candidate"
cv_cache = {}

def match_cv_with_criteria(cv_text, criteria_json):
    # Generate hash for the CV
    cv_hash = generate_cv_hash(cv_text)

    # Check if the CV has already been processed
    if cv_hash in cv_cache:
        return cv_cache[cv_hash]

    # Validate input
    if not criteria_json:
        st.error("Criteria JSON is empty or invalid.")
        return {'cv_text': cv_text}

    try:
        # Extract candidate name from CV text
        candidate_name = extract_candidate_name_from_cv(cv_text)

        # Load criteria JSON
        criteria = json.loads(criteria_json)

        # Extract total years of experience
        experience_info = extract_experience_from_cv(cv_text)
        total_years = (experience_info.get("total_years", 0))
        print("total:",total_years)
        try:
            total_years = float(total_years)  # Convert to float if possible
        except ValueError:
            total_years = 0
        # Extract required years of experience
        required_experience = extract_required_experience(criteria.get("experience", "0"))

        # Prepare detailed GPT prompt for matching
        prompt = (
        "CRITICAL INSTRUCTIONS: EVALUATE STRICTLY AGAINST ESSENTIAL CRITERIA ONLY\n\n"
        "Evaluation Rules:\n"
        "1. ONLY match against ESSENTIAL CRITERIA from job description\n"
        "2. COMPLETELY IGNORE Desirable Criteria\n"
        "3. Candidate MUST PASS if ALL Essential Criteria are met\n"
        "4. EXCLUDE any requirements from Desirable Criteria in 'Missing Requirements'\n\n"
        "Essential Criteria Specifics:\n"
        "- Education: B.E / B.Tech (Any Stream) / MCA / MSc (CS/IT) or post-graduation in (CS/IT)\n"
        "- Experience: Minimum 8 years total\n"
        "- Must have minimum 2 web-based development assignments\n\n"
        "Provide JSON response:\n"
        "{\n"
        "  \"Matching Education\": [List of matched ESSENTIAL education qualifications],\n"
        "  \"Matching Experience\": [List of matched ESSENTIAL work experiences],\n"
        "  \"Matching Skills\": [List of matched ESSENTIAL skills],\n"
        "  \"Matching Certifications\": [list of matching certification from essential criteria Only],\n"
        "  \"Missing Requirements\": [ONLY missing ESSENTIAL requirements],\n"
        "  \"Skill Stratification\": {\"skill1\": score, \"skill2\": score},\n"
        "  \"Pass\": true/false based STRICTLY on ESSENTIAL criteria\n"
        "}\n\n"
        "Job Description Essential Criteria:\n"
        f"{criteria_json}\n\n"
        "Candidate CV:\n"
        f"{cv_text}\n\n"
        f"Total Years of Experience: {total_years} Years\n\n"
        "IMPORTANT: Strictly evaluate ONLY against ESSENTIAL CRITERIA. DO NOT include Desirable Criteria requirements."
    )


        # Fetch GPT-3.5 analysis
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a precise job criteria matcher. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0
        )

        # Extract and clean the response
        raw_response = response.choices[0].message.content.strip()
        
        # Additional JSON parsing safeguards
        try:
            matching_results = json.loads(raw_response)
        except json.JSONDecodeError:
            # Try to extract JSON between first { and last }
            import re
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                try:
                    matching_results = json.loads(json_match.group(0))
                except:
                    # Fallback to default results
                    matching_results = {
                        "Matching Education": [],
                        "Matching Experience": [],
                        "Matching Skills": [],
                        "Matching Certifications": [],
                        "Missing Requirements": [],
                        "Skill Stratification": {},
                        "Pass": False
                    }
            else:
                raise ValueError("Unable to parse JSON response")
            
        # Prepare final results
        results = {
            "Candidate Name": candidate_name,
            "Status": "Pass" if matching_results.get('Pass', False) else "Fail",
            "Total Years of Experience": total_years,
            "Matching Education": matching_results.get('Matching Education', []),
            "Matching Experience": matching_results.get('Matching Experience', []),
            "Matching Skills": matching_results.get('Matching Skills', []),
            "Matching Certifications": matching_results.get('Matching Certifications', []),
            "Missing Requirements": matching_results.get('Missing Requirements', []),
            "Stratification": matching_results.get('Skill Stratification', {}),
            "Skill Score": sum(matching_results.get('Skill Stratification', {}).values())
        }
        #print("results",results)
        # Cache results
        cv_cache[cv_hash] = results

        return results

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Return a default result structure
        return {
            "Candidate Name": candidate_name or "Unknown Candidate",
            "Status": "Fail",
            "Total Years of Experience": 0,
            "Matching Education": [],
            "Matching Experience": [],
            "Matching Skills": [],
            "Matching Certifications": [],
            "Missing Requirements": ["Failed to process CV"],
            "Stratification": {},
            "Skill Score": 0
        }

def extract_required_experience(experience_str):
    """
    Extract numeric value(s) for required experience from a given string.
    
    Args:
    - experience_str (str): The string containing required experience information.

    Returns:
    - float: The extracted numeric value for required experience.
    """
    try:
        # Use regex to find all numbers in the string and convert them to floats
        numbers = re.findall(r'\d+', experience_str)
        
        # Convert found numbers to floats and return the maximum value found (assuming multiple values)
        return max(float(num) for num in numbers) if numbers else 0.0
        
    except Exception as e:
        st.warning(f"Couldn't extract a numeric value from required experience: {experience_str}. Error: {e}")
        return 0.0  # Default to 0 if extraction fails
    
def get_skill_score_justification(criteria_json, skill, score, cv_text):
    """
    Generate a justification for the skill score based on the candidate's resume text
    and evaluation criteria.
    
    :param criteria_json: JSON object with evaluation criteria
    :param skill: Skill being evaluated
    :param score: Score assigned to the skill
    :param cv_text: Candidate's resume text
    :return: Justification for the skill score
    """
    # Prepare the prompt
    if score == 0:
        prompt = (
            f"Based on the evaluation criteria provided in {criteria_json}, explain in 2 bullet points only,explain every bullent points just 20 words only why the candidate's "
            f"resume text '{cv_text}' does not demonstrate the necessary skills for '{skill}', resulting in a score of {score}/10. "
            "Focus on specific, unique shortcomings in the candidate's resume text that justify this score. "
            "Avoid generic or repetitive explanations across different resumes."
            "*Strictly Every Bullent Points are started from next line.*"
        )
    else:
        prompt = (
            f"Based on the evaluation criteria provided in {criteria_json}, provide a concise justification in 2 bullet points only,explain every bullent points just 20 words only "
            f"explaining why the candidate's resume text '{cv_text}' demonstrates a match for the skill '{skill}' with a score of {score}/10. "
            "Focus on specific, unique strengths in the candidate's resume text that align with the criteria and justify the score. "
            "Avoid reusing the same points for different resumes and ensure each explanation is unique."
            "*Strictly Every Bullent Points are started from next line.*"
        )

    # Using the 'openai.chat.completions.create' method
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100,
        temperature=0.1
    )

    # Access the explanation from the response
    explanation = response.choices[0].message.content.strip()

    # Return the explanation
    return explanation

def display_pass_fail_verdict(results, cv_text):
    candidate_name = results['Candidate Name']
    skill_scores = results.get("Stratification", {})

    # Debugging: Print skill_scores to verify the data
    logging.debug(f"Skill scores for {candidate_name}: {skill_scores}")

    with st.container():
        # Pass/Fail verdict
        pass_fail = results.get("Status", "Fail")
        if pass_fail == 'Pass':
            st.markdown(f"<h2 style='color: green; font-size: 1em;'>🟢 Profile Eligibility: PASS ✔️</h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: red; font-size: 1em;'>🔴 Profile Eligibility: FAIL ❌</h2>", unsafe_allow_html=True)

        # Check for skill scores
        if skill_scores:
            # Create an expander for candidate selection
            expander = st.expander(f"**Click to view {candidate_name}'s detailed skill assessment**")

            with expander:
                # Prepare a table with candidate name, skill, score, and justification
                table_data = []
                x=0
                for skill, score in skill_scores.items():
                    
                    # Generate justification dynamically for each skill
                    explanation = get_skill_score_justification(criteria_json, skill, score, cv_text)
                    x=x+10
                    # Append the row to the table data
                    table_data.append({
                        "Candidate Name": candidate_name,
                        "Skill": skill,
                        "Score": f"{score}/10",
                        "Justification": explanation
                    })
                   
                df = pd.DataFrame(table_data)
                blankIndex=[''] * len(df)
                df.index=blankIndex
                #st.markdown(df.to_html(index=False), unsafe_allow_html=True)
                st.table(df)
                total_possible_score = x
        # Display overall skill score
                Additional_Skill_Score = round(results.get("Skill Score", 0), 1)
                st.markdown(f"""
            <h3 style='font-size:20px;'>Additional_Skill_Score: <strong>{Additional_Skill_Score:.1f}</strong> out of {total_possible_score}</h3>
            """, unsafe_allow_html=True)

        # Pass/Fail message based on the final status
        if pass_fail == "Pass":
            st.success("The candidate has passed based on the job description criteria.")
        else:
            st.error("The candidate has failed to meet the job description criteria.")

        st.markdown("</div>", unsafe_allow_html=True)


# Function to display and rank candidates in a table
def display_candidates_table(candidates):
    if not candidates:
        st.info("No candidates to display.")
        return

    # Standardize the results dictionary
    processed_candidates = []
    for candidate in candidates:
        processed_candidate = {
            'Candidate Name': candidate.get('Candidate Name', 'Unknown Candidate'),
            'Status': candidate.get('Status', 'Fail'),
            'Total Years of Experience': candidate.get('Total Years of Experience', 0),
            'Skill Score': candidate.get('Skill Score', 0),
            'Matching Education': ', '.join([str(item) for item in candidate.get('Matching Education', [])]) or 'None',
            'Matching Experience': ', '.join([str(item) for item in candidate.get('Matching Experience', [])]) or 'None',
            'Matching Skills': ', '.join([str(item) for item in candidate.get('Matching Skills', [])]) or 'None',
            'Matching Certifications': ', '.join([str(item) for item in candidate.get('Matching Certifications', [])]) or 'None',
            'Missing Requirements': ', '.join([str(item) for item in candidate.get('Missing Requirements', [])]) or 'None',

            'Stratification': str(candidate.get('Stratification', {}))
        }
        processed_candidates.append(processed_candidate)

    # Create DataFrame
    df = pd.DataFrame(processed_candidates)

    # Ensure columns are properly formatted
    # Ensure columns are properly formatted
    df['Total Years of Experience'] = df['Total Years of Experience'].apply(lambda x: float(x)).round(1)
    df['Skill Score'] = df['Skill Score'].apply(lambda x: float(x)).round(0)
    print("skilll score",df['Skill Score'])
    # Format 'Total Years of Experience' for display
    df['Total Years of Experience'] = df['Total Years of Experience'].map('{:.1f}'.format)
    

    # Sorting logic
    df['rank'] = df.apply(lambda row: (
        0 if row['Status'] == 'Pass' else 1, 
        -row['Skill Score'], 
        -float(row['Total Years of Experience'])
    ), axis=1)
    df = df.sort_values(by='rank').drop(columns=['rank'])


    # Reorder columns for better readability
    column_order = [
        'Candidate Name', 'Status', 'Total Years of Experience', 
        'Matching Education', 'Matching Experience', 'Matching Skills', 
        'Matching Certifications', 'Missing Requirements', 
        'Skill Score', 'Stratification'
    ]
    df = df[column_order]
    
    # Display the table
    st.markdown("## :trophy: Candidate Rankings")
    
    # Styling
    def color_pass_fail(val):
        color = 'lightgreen' if val == 'Pass' else 'lightcoral'
        return f'background-color: {color}'

    # Apply conditional formatting only to the 'Status' column
    styled_df = (
        df.style
        .applymap(color_pass_fail, subset=['Status'])  # Apply styling to 'Status' column
        .format(precision=1)  # Set formatting for numeric columns
    )

# Display styled DataFrame
    st.dataframe(styled_df, hide_index=True)

    return df

# Add a styled box for the file uploaders
st.markdown("""
    <div style="background-color: lightblue; padding: 4px; border-radius: 5px; text-align: left; 
                 max-width: 380px;">
        <p style="color: black; margin-left: 60px; margin-top: 20px; font-weight: bold;font-size: 20px;">Upload Job Description (PDF)</p>
    </div>
""", unsafe_allow_html=True)

# File uploader for Job Description
jd_file = st.file_uploader(" ", type=["pdf", "docx","doc"])

# Header for Candidate Resumes Upload
st.markdown("""
    <div style="background-color: lightblue; padding: 2px; border-radius: 5px; text-align: left; 
                 max-width: 400px;">
        <p style="color: black; margin-left: 55px; margin-top: 20px; font-weight: bold;font-size: 20px;">Upload Candidate Resumes (PDF)</p>
    </div>
""", unsafe_allow_html=True)

# File uploader for Candidate Resumes
cv_files = st.file_uploader("", type=["pdf", "docx","doc"], accept_multiple_files=True)
st.write(f"Total CV'S Uploaded: {len(cv_files)} ")
# Ensure criteria_json is initialized in session state
if 'criteria_json' not in st.session_state:
    st.session_state['criteria_json'] = None

# Button to extract criteria and match candidates
if st.button("Extract Criteria and Match Candidates"):
    if jd_file:
        jd_text = extract_text_from_uploaded_pdf(jd_file)
        #print("jdtext:",jd_text)
        if jd_text:
            criteria_json = use_genai_to_extract_criteria(jd_text)
            #print("criteria jd text",criteria_json)
            if criteria_json:
                # Save criteria in session state
                st.session_state.criteria_json = criteria_json
                st.success("Job description criteria extracted successfully.")

                # Proceed to match candidates
                if cv_files:
                    candidates_results = []
                    for cv_file in cv_files:
                        cv_text = extract_text_from_uploaded_pdf(cv_file)
                        if cv_text:
                            results = match_cv_with_criteria(cv_text, st.session_state.criteria_json)
                            if results:
                                candidates_results.append(results)
                        else:
                            st.error(f"Failed to extract text from CV file.")
                    
                    # Display candidates table first
                    if candidates_results:
                        display_candidates_table(candidates_results)
                        
                    # Then display individual matching details
                    for result in candidates_results:
                        st.markdown(f"### Results for {result['Candidate Name']}:")
                        display_pass_fail_verdict(result, cv_text)
                else:
                    st.error("Please upload at least one CV PDF.")
            else:
                st.error("Failed to extract job description criteria.")
        else:
            st.error("The uploaded JD file appears to be empty.")
    else:
        st.error("Please upload a Job Description PDF.")


disclaimer_text = """
<div style="color: grey; margin-top: 50px;">
    <strong>Disclaimer:</strong> Certification validity must be verified manually. 
    These results are AI-generated, and candidates should be thoroughly evaluated 
    through technical interviews For the information provided in their resumes. 
    
</div>
"""

st.markdown(disclaimer_text, unsafe_allow_html=True)

     
        
footer = """
    <style>
    body {
        margin: 0;
        padding-top: 70px;  /* Add padding to prevent content from being hidden behind the footer */
    }
    .footer {
        position: absolute;
        top: 80px;
        left: 0;
        width: 100%;
        background-color: #002F74;
        color: white;
        text-align: center;
        padding: 5px;
        font-weight: bold;
        z-index: 1000;  /* Ensure it is on top of other elements */
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
    }
    .footer p {
        font-style: italic;
        font-size: 14px;
        margin: 0;
        flex: 1 1 50%;  /* Flex-grow, flex-shrink, flex-basis */
    }
    @media (max-width: 600px) {
        .footer p {
            flex-basis: 100%;
            text-align: center;
            padding-top: 10px;
        }
    }
    </style>
    <div class="footer">
        <p style="text-align: left;">Copyright © 2024 MPSeDC. All rights reserved.</p>
        <p style="text-align: right;">The responses provided on this website are AI-generated. User discretion is advised.</p>
    </div>
"""
