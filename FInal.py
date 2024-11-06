#Working prototype for demo- 06/11/2024
#Working prototype for demo- 06/11/2024
import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import tempfile
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas
from PIL import Image as PILImage
import json
import re
import io
from pdf2docx import Converter
import base64

# Load environment variables
load_dotenv()
genai_api_key = os.getenv("GOOGLE_API_KEY")


@st.cache_resource
def get_pdf_text(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def extract_info_with_gemini(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)

    prompt = f"""
    You are a highly capable document analysis assistant. Given a resume text, extract and return the following information in JSON format:
    {{
        "personal_info": {{
            "name": "",
            "email": "",
            "phone": "",
            "address": "",
            "LinkedIn": ""
        }},
        "professional_summary": "",
        "education": [
            {{
                "degree": "",
                "institution": "",
                "year": "",
                "details": ""
            }}
        ],
        "experience": [
            {{
                "title": "",
                "company": "",
                "duration": "",
                "responsibilities": []
            }}
        ],
        "projects": [
            {{
                "name": "",
                "description": "",
                "technologies": ""
            }}
        ],
        "skills": {{
            "technical": [],
            "soft": []
        }},
        "courses_and_certifications": [
            {{
                "name": "",
                "issuer": "",
                "year": "",
                "type": ""  # Either "course" or "certification"
            }}
        ]
    }}

    Resume Text: {text}

    Please follow these guidelines:
    1. Place only valid LinkedIn URLs in "LinkedIn".

    Return only the JSON object with the extracted information, maintaining the exact structure shown above.
    """

    response = model.predict(prompt)
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            
            # Validate LinkedIn URL
            linkedin_url = data['personal_info'].get('LinkedIn', '')
            if linkedin_url and not re.match(r'https?://(www\.)?linkedin\.com/', linkedin_url):
                # Move non-LinkedIn URL to 'other_links'
                data['personal_info'].setdefault('other_links', []).append(linkedin_url)
                data['personal_info']['LinkedIn'] = ""

            return data
        except json.JSONDecodeError:
            return None
    return None


@st.cache_resource
def extract_images_from_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    images = []
    for page in pdf_reader.pages:
        if "/XObject" in page["/Resources"]:
            x_objects = page["/Resources"]["/XObject"].get_object()
            for obj in x_objects:
                if x_objects[obj]["/Subtype"] == "/Image":
                    image_data = x_objects[obj].get_data()
                    image = PILImage.open(io.BytesIO(image_data))
                    images.append(image)
    return images

class ResumeCanvas:
    def __init__(self, pagesize=letter):
        self.pages = []
        self.current_page = None
        self.packet = None
        self.pagesize = pagesize
        self.width, self.height = pagesize
        # Significantly reduced margins to maximize space usage
        self.left_margin = 0.75 * inch  # Reduced left margin
        self.right_margin = 0.03 * inch  # Reduced right margin
        self.top_margin = 0.75 * inch    # Reduced top margin
        self.bottom_margin = 1 * inch  # Reduced bottom margin
        self.y = self.height - self.top_margin
        self.line_spacing = 2  # Reduced line spacing
        self.new_page()

    def new_page(self):
        if self.packet:
            self.current_page.save()
            self.pages.append(self.packet)

        self.packet = io.BytesIO()
        self.current_page = canvas.Canvas(self.packet, pagesize=self.pagesize)
        self.y = self.height - self.top_margin

    def check_space(self, needed_space):
        if self.y - needed_space < self.bottom_margin:
            self.new_page()
            return True
        return False

    def draw_text(self, text, font_name, font_size, indent=0):
        # Calculate maximum available width
        available_width = self.width - self.left_margin - self.right_margin - indent
        
        # More aggressive text wrapping calculation
        avg_char_width = font_size * 0.55  # Slightly reduced character width estimate
        
        if len(text) * avg_char_width > available_width:
            words = text.split()
            line = ""
            for word in words:
                test_line = line + word + " "
                if len(test_line.strip()) * avg_char_width < available_width:
                    line = test_line
                else:
                    if line.strip():  # Only draw if there's content
                        self.check_space(font_size + self.line_spacing)
                        self.current_page.setFont(font_name, font_size)
                        self.current_page.drawString(self.left_margin + indent, self.y, line.strip())
                        self.y -= font_size + self.line_spacing
                    line = word + " "
            if line.strip():  # Only draw if there's content
                self.check_space(font_size + self.line_spacing)
                self.current_page.setFont(font_name, font_size)
                self.current_page.drawString(self.left_margin + indent, self.y, line.strip())
                self.y -= font_size + self.line_spacing
        else:
            if text.strip():  # Only draw if there's content
                self.check_space(font_size + self.line_spacing)
                self.current_page.setFont(font_name, font_size)
                self.current_page.drawString(self.left_margin + indent, self.y, text)
                self.y -= font_size + self.line_spacing

    def draw_section(self, title, content_list, font_size=12, section_spacing=8):
        # Reduced section spacing
        self.y -= section_spacing
        
        # Draw section title using more space
        if title:
            self.current_page.setFont("Times-Bold", 12)
            self.current_page.drawString(self.left_margin, self.y, title)
            self.y -= 14  # Reduced spacing after title
        
        # Process content with minimal spacing
        for item in content_list:
            if isinstance(item, str) and item.strip():
                self.draw_text(item, "Times-Roman", font_size)
            elif isinstance(item, dict):
                for key, value in item.items():
                    if isinstance(value, list):
                        for bullet in value:
                            # Minimal indent for bullets
                            self.draw_text(f"• {bullet}", "Times-Roman", font_size, indent=0.15 * inch)
                    else:
                        # Draw headers with minimal spacing
                        if value.strip():
                            self.current_page.setFont("Times-Bold", font_size)
                            self.current_page.drawString(self.left_margin, self.y, value)
                            self.y -= font_size + self.line_spacing
        
        # Minimal spacing between sections
        self.y -= section_spacing

    def draw_personal_info(self, name, contact_info):
        # Maximize space for header section
        original_y = self.y
        
        # Draw name in larger font
        self.current_page.setFont("Times-Bold", 16)
        self.current_page.drawString(self.left_margin, self.y, name)
        self.y -= 20
        
        # Draw contact info in smaller font with minimal spacing
        self.current_page.setFont("Times-Roman", 12)
        x_pos = self.left_margin
        
        for info in contact_info:
            if info.strip():
                text_width = self.current_page.stringWidth(info, "Times-Roman", 12)
                if x_pos + text_width > (self.width - self.right_margin):
                    self.y -= 12
                    x_pos = self.left_margin
                self.current_page.drawString(x_pos, self.y, info)
                x_pos += text_width + 20  # Minimal spacing between contact items
        
        self.y -= 15  # Minimal spacing after contact info

    def draw_image(self, image, x, y, width, height):
        self.current_page.drawImage(image, x, y, width=width, height=height)

    def get_all_pages(self):
        self.current_page.save()
        self.pages.append(self.packet)
        return self.pages
def create_content_pdf(data, profile_picture=None):
    canvas_handler = ResumeCanvas()
    
    initial_y = canvas_handler.height - 1 * inch  # Set the initial top position for content

    # Profile Picture handling - Now aligned to the right with a slight downward offset
    if profile_picture:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_image_file:
            profile_picture.save(tmp_image_file, format="PNG")
            tmp_image_path = tmp_image_file.name
        
        picture_width = 1 * inch
        picture_height = 1 * inch
        picture_x = canvas_handler.width - picture_width - canvas_handler.right_margin - 0.5 * inch  # Align slightly left of right margin
        picture_y = initial_y - 0.3 * inch  # Push the image down by 0.3 inches
        
        # Draw the image at the adjusted position
        canvas_handler.draw_image(tmp_image_path, x=picture_x, y=picture_y, 
                                  width=picture_width, height=picture_height)

    # Keep content aligned at the original `initial_y` without the downward offset
    canvas_handler.y = initial_y
    
    # Personal Information (left-aligned)
    canvas_handler.current_page.setFont("Times-Bold", 16)
    canvas_handler.current_page.drawString(canvas_handler.left_margin, canvas_handler.y, 
                                           data['personal_info']['name'])
    
    # Contact Information
    canvas_handler.y -= 20
    canvas_handler.current_page.setFont("Times-Roman", 12)
    contact_info = []
    if data['personal_info']['address']:
        contact_info.append(data['personal_info']['address'])
    if data['personal_info']['LinkedIn']:
        contact_info.append(f"LinkedIn: {data['personal_info']['LinkedIn']}")
    
    # Left-align contact information
    for info in contact_info:
        canvas_handler.current_page.drawString(canvas_handler.left_margin, canvas_handler.y, info)
        canvas_handler.y -= 15

    canvas_handler.y -= 10  # Add extra space before starting sections

    # Professional Summary
    if data['professional_summary']:
        canvas_handler.draw_section("PROFESSIONAL SUMMARY", [data['professional_summary']])

    # Education
    education_items = []
    for edu in data['education']:
        education_items.extend([
            f"{edu['degree']} - {edu['institution']}",
            f"{edu['year']}",
            edu['details'] if edu['details'] else ""
        ])
    canvas_handler.draw_section("EDUCATION", education_items)

    # Skills
    skills_items = [
        f"Technical Skills: {', '.join(data['skills']['technical'])}",
    ]
    if data['skills']['soft']:
        skills_items.append(f"Soft Skills: {', '.join(data['skills']['soft'])}")
    canvas_handler.draw_section("SKILLS", skills_items)

    # Experience
    experience_items = []
    for exp in data['experience']:
        experience_items.append({
            'title': f"{exp['title']} - {exp['company']}",
            'duration': exp['duration'],
            'responsibilities': exp['responsibilities']
        })
    canvas_handler.draw_section("PROFESSIONAL EXPERIENCE", experience_items)

    # Projects
    project_items = []
    for project in data['projects']:
        project_items.extend([
            project['name'],
            project['description'],
            f"Technologies: {project['technologies']}" if project['technologies'] else ""
        ])
    canvas_handler.draw_section("PROJECTS", project_items)

    # Courses and Certifications
    if data['courses_and_certifications']:
        cert_items = [
            f"{item['name']} - {item['issuer']} ({item['year']}) - {item['type'].capitalize()}"
            for item in data['courses_and_certifications']
        ]
        canvas_handler.draw_section("COURSES AND CERTIFICATIONS", cert_items)

    # Cleanup
    if profile_picture:
        os.remove(tmp_image_path)

    return canvas_handler.get_all_pages()



def merge_with_template(content_pdfs, template_path, output_path):
    """Merge the content PDFs with the template PDF, maintaining template design across all pages"""
    template = PdfReader(template_path)
    output = PdfWriter()

    # Get the template page
    template_page = template.pages[0]

    # Process each content page
    for page_num, content_pdf in enumerate(content_pdfs):
        content_pdf.seek(0)
        content = PdfReader(content_pdf)

        # Create a new copy of the template page for each content page
        # This ensures we have a fresh template for each page
        new_template = PdfReader(template_path)
        current_template = new_template.pages[0]

        # Merge the content with the template
        current_template.merge_page(content.pages[0])
        output.add_page(current_template)

    # Write the merged PDF to file
    with open(output_path, "wb") as output_file:
        output.write(output_file)
        
def pdf_to_word(pdf_path, word_output_path):
    cv = Converter(pdf_path)
    cv.convert(word_output_path, start=0, end=None)  # Convert all pages
    cv.close()
    return word_output_path

def process_document(file, template_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + file.name.split('.')[-1]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        if file.name.split('.')[-1].lower() == 'pdf':
            text = get_pdf_text(tmp_file_path)
        else:
            raise ValueError(f"Unsupported file type")

        # Extract structured information
        extracted_info = extract_info_with_gemini(text)

        if extracted_info:
            # Get the person's name and create filenames
            name = extracted_info['personal_info']['name']
            clean_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
            output_pdf_path = f"{clean_name}_resume.pdf"
            output_word_path = f"{clean_name}_resume.docx"

            # Create content PDFs
            content_pdfs = create_content_pdf(extracted_info)

            # Merge with template
            merge_with_template(content_pdfs, template_path, output_pdf_path)

            # Convert the PDF to a Word document
            pdf_to_word(output_pdf_path, output_word_path)

            return extracted_info, output_pdf_path, output_word_path
        return None, None, None
    finally:
        os.unlink(tmp_file_path)


def process_document(file, template_path):
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + file.name.split('.')[-1]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        if file.name.split('.')[-1].lower() == 'pdf':
            text = get_pdf_text(tmp_file_path)
            images = extract_images_from_pdf(tmp_file_path)  # Extract images from PDF

        else:
            raise ValueError(f"Unsupported file type")

        # Extract structured information
        extracted_info = extract_info_with_gemini(text)

        if extracted_info:
            # Get the person's name and create filenames
            name = extracted_info['personal_info']['name']
            clean_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
            output_pdf_path = f"{clean_name}_resume.pdf"
            output_word_path = f"{clean_name}_resume.docx"

            # Assume the first extracted image is the profile picture (if available)
            profile_picture = images[0] if images else None

            # Create content PDFs
            content_pdfs = create_content_pdf(extracted_info, profile_picture)

            # Merge with template
            merge_with_template(content_pdfs, template_path, output_pdf_path)

            # Convert the PDF to a Word document
            pdf_to_word(output_pdf_path, output_word_path)

            return extracted_info, output_pdf_path, output_word_path
        return None, None, None
    finally:
        os.unlink(tmp_file_path)
# Previous functions remain the same...
# [Include all your existing functions here without changes]
def create_content_pdf_no_picture(data):
    """
    Creates PDF content without profile picture
    """
    canvas_handler = ResumeCanvas()
    
    initial_y = canvas_handler.height - 1 * inch

    # Personal Information (left-aligned)
    canvas_handler.current_page.setFont("Times-Bold", 16)
    canvas_handler.current_page.drawString(canvas_handler.left_margin, canvas_handler.y, 
                                           data['personal_info']['name'])
    
    # Contact Information
    canvas_handler.y -= 20
    canvas_handler.current_page.setFont("Times-Roman", 12)
    contact_info = []
    if data['personal_info']['address']:
        contact_info.append(data['personal_info']['address'])
    if data['personal_info']['LinkedIn']:
        contact_info.append(f"LinkedIn: {data['personal_info']['LinkedIn']}")
    
    # Left-align contact information
    for info in contact_info:
        canvas_handler.current_page.drawString(canvas_handler.left_margin, canvas_handler.y, info)
        canvas_handler.y -= 15

    canvas_handler.y -= 10

    # [Rest of the sections remain the same as in create_content_pdf]
    if data['professional_summary']:
        canvas_handler.draw_section("PROFESSIONAL SUMMARY", [data['professional_summary']])

    education_items = []
    for edu in data['education']:
        education_items.extend([
            f"{edu['degree']} - {edu['institution']}",
            f"{edu['year']}",
            edu['details'] if edu['details'] else ""
        ])
    canvas_handler.draw_section("EDUCATION", education_items)

    skills_items = [
        f"Technical Skills: {', '.join(data['skills']['technical'])}",
    ]
    if data['skills']['soft']:
        skills_items.append(f"Soft Skills: {', '.join(data['skills']['soft'])}")
    canvas_handler.draw_section("SKILLS", skills_items)

    experience_items = []
    for exp in data['experience']:
        experience_items.append({
            'title': f"{exp['title']} - {exp['company']}",
            'duration': exp['duration'],
            'responsibilities': exp['responsibilities']
        })
    canvas_handler.draw_section("PROFESSIONAL EXPERIENCE", experience_items)

    project_items = []
    for project in data['projects']:
        project_items.extend([
            project['name'],
            project['description'],
            f"Technologies: {project['technologies']}" if project['technologies'] else ""
        ])
    canvas_handler.draw_section("PROJECTS", project_items)

    if data['courses_and_certifications']:
        cert_items = [
            f"{item['name']} - {item['issuer']} ({item['year']}) - {item['type'].capitalize()}"
            for item in data['courses_and_certifications']
        ]
        canvas_handler.draw_section("COURSES AND CERTIFICATIONS", cert_items)

    return canvas_handler.get_all_pages()

def process_document_no_picture(file, template_path):
    """
    Process document without including profile picture
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + file.name.split('.')[-1]) as tmp_file:
        tmp_file.write(file.getvalue())
        tmp_file_path = tmp_file.name

    try:
        if file.name.split('.')[-1].lower() == 'pdf':
            text = get_pdf_text(tmp_file_path)
        else:
            raise ValueError(f"Unsupported file type")

        extracted_info = extract_info_with_gemini(text)

        if extracted_info:
            name = extracted_info['personal_info']['name']
            clean_name = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')
            output_pdf_path = f"{clean_name}_resume_no_picture.pdf"
            output_word_path = f"{clean_name}_resume_no_picture.docx"

            # Create content PDFs without profile picture
            content_pdfs = create_content_pdf_no_picture(extracted_info)

            # Merge with template
            merge_with_template(content_pdfs, template_path, output_pdf_path)

            # Convert to Word
            pdf_to_word(output_pdf_path, output_word_path)

            return output_pdf_path, output_word_path
        return None, None
    finally:
        os.unlink(tmp_file_path)


def display_pdf(pdf_path):
    """
    Display a PDF file in the Streamlit app
    """
    with open(pdf_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    
    # Embed PDF viewer
    pdf_display = f"""
        <iframe
            src="data:application/pdf;base64,{base64_pdf}"
            width="100%"
            height="800px"
            type="application/pdf">
        </iframe>
    """
    st.markdown(pdf_display, unsafe_allow_html=True)
def save_uploadedfile(uploadedfile):
    """
    Save uploaded file to temp directory and return the path
    """
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, uploadedfile.name)
    with open(temp_path, "wb") as f:
        f.write(uploadedfile.getvalue())
    return temp_path

def main():
    st.set_page_config(page_title="Resume Formatter", page_icon="📄", layout="wide")
    st.header("Resume Formatter and Analyzer 📄")
    st.subheader("Upload your PDF resume to get a professionally formatted version")

    # Initialize session state variables
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = {}
    if 'clear_clicked' not in st.session_state:
        st.session_state.clear_clicked = False

    def clear_state():
        st.session_state.clear_clicked = True
        st.session_state.processed_files = {}

    template_path = "Format.pdf"
    if not os.path.exists(template_path):
        st.error("Template file 'Format.pdf' not found. Please ensure it exists in the same directory as this script.")
        return

    if st.button("Clear and Upload New Resume"):
        clear_state()
        st.rerun()

    uploaded_file = st.file_uploader("Upload your PDF Document", type=["pdf"], key="pdf_uploader")

    if uploaded_file:
        if st.session_state.clear_clicked:
            st.session_state.clear_clicked = False

        file_identifier = f"{uploaded_file.name}_{os.path.getsize(save_uploadedfile(uploaded_file))}"

        if file_identifier not in st.session_state.processed_files:
            with st.spinner("Processing and formatting the resume..."):
                # Process with profile picture
                extracted_info, output_pdf_path, output_word_path = process_document(uploaded_file, template_path)
                
                # Process without profile picture
                output_pdf_no_pic, output_word_no_pic = process_document_no_picture(uploaded_file, template_path)

                if all([extracted_info, output_pdf_path, output_word_path, output_pdf_no_pic, output_word_no_pic]):
                    # Store all results in session state
                    st.session_state.processed_files[file_identifier] = {
                        'extracted_info': extracted_info,
                        'pdf_content': open(output_pdf_path, "rb").read(),
                        'word_content': open(output_word_path, "rb").read(),
                        'pdf_path': output_pdf_path,
                        'word_path': output_word_path,
                        'pdf_no_pic_content': open(output_pdf_no_pic, "rb").read(),
                        'word_no_pic_content': open(output_word_no_pic, "rb").read(),
                        'pdf_no_pic_path': output_pdf_no_pic,
                        'word_no_pic_path': output_word_no_pic
                    }
                else:
                    st.error("Failed to process the resume. Please try again.")
                    return

        results = st.session_state.processed_files[file_identifier]
        
        st.success("Resume processed and formatted successfully!")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Preview")
            display_pdf(results['pdf_path'])

        with col2:
            st.subheader("Extracted Information")
            st.json(results['extracted_info'])

            st.subheader("Download Options")
            
            # Version with profile picture
            st.markdown("**With Profile Picture:**")
            st.download_button(
                label="Download PDF (with picture)",
                data=results['pdf_content'],
                file_name=f"{uploaded_file.name.split('.')[0]}_formatted.pdf",
                mime="application/pdf",
                key="pdf_download"
            )

            st.download_button(
                label="Download Word (with picture)",
                data=results['word_content'],
                file_name=f"{uploaded_file.name.split('.')[0]}_formatted.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="word_download"
            )

            # Version without profile picture
            st.markdown("**Without Profile Picture:**")
            st.download_button(
                label="Download PDF (no picture)",
                data=results['pdf_no_pic_content'],
                file_name=f"{uploaded_file.name.split('.')[0]}_formatted_no_picture.pdf",
                mime="application/pdf",
                key="pdf_download_no_pic"
            )

            st.download_button(
                label="Download Word (no picture)",
                data=results['word_no_pic_content'],
                file_name=f"{uploaded_file.name.split('.')[0]}_formatted_no_picture.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                key="word_download_no_pic"
            )

if __name__ == "__main__":
    main()




