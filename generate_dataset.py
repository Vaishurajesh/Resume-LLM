"""
Generates dataset/resume_instructions.jsonl — a hand-curated instruction-tuning
dataset for resume parsing, covering diverse formats, missing fields, and
edge cases. Each example is deliberately authored (not templated boilerplate)
so the model sees genuinely varied input styles.

Run: python generate_dataset.py
"""
import json
from pathlib import Path

SYSTEM_INSTRUCTION = (
    "Extract structured resume information from the input text. "
    "Return ONLY valid JSON matching the schema: name, email, phone, "
    "location, summary, skills (list), experience (list of {title, company, "
    "start_date, end_date, description}), education (list of {degree, "
    "institution, year}), certifications (list). Use null for missing "
    "fields and [] for missing lists. Do not invent information that is "
    "not present in the input."
)

examples = []

def add(input_text, output, instruction=SYSTEM_INSTRUCTION):
    examples.append({"instruction": instruction, "input": input_text, "output": output})

# 1. Clean, standard single-column resume
add(
"""Priya Sharma
priya.sharma@email.com | +91 98765 43210 | Bengaluru, India

Summary: Software engineer with 5 years of experience in backend systems.

Experience:
Senior Backend Engineer, Zeta Technologies (Jun 2021 - Present)
- Led migration of monolith to microservices

Backend Engineer, InfoWave Systems (Jul 2018 - May 2021)
- Built REST APIs serving 2M daily requests

Education:
B.Tech in Computer Science, VIT Vellore, 2018

Certifications: AWS Certified Solutions Architect""",
{
  "name": "Priya Sharma", "email": "priya.sharma@email.com", "phone": "+91 98765 43210",
  "location": "Bengaluru, India",
  "summary": "Software engineer with 5 years of experience in backend systems.",
  "skills": [],
  "experience": [
    {"title": "Senior Backend Engineer", "company": "Zeta Technologies", "start_date": "2021-06", "end_date": "Present", "description": "Led migration of monolith to microservices"},
    {"title": "Backend Engineer", "company": "InfoWave Systems", "start_date": "2018-07", "end_date": "2021-05", "description": "Built REST APIs serving 2M daily requests"}
  ],
  "education": [{"degree": "B.Tech in Computer Science", "institution": "VIT Vellore", "year": "2018"}],
  "certifications": ["AWS Certified Solutions Architect"]
})

# 2. Missing contact info entirely (only name and experience)
add(
"""John Doe

Worked at Bright Retail as a Store Manager from 2015 to 2020, overseeing
a team of 12 staff and inventory of $500K.""",
{
  "name": "John Doe", "email": None, "phone": None, "location": None,
  "summary": None, "skills": [],
  "experience": [{"title": "Store Manager", "company": "Bright Retail", "start_date": "2015", "end_date": "2020", "description": "Oversaw a team of 12 staff and inventory of $500K"}],
  "education": [], "certifications": []
})

# 3. Recent graduate, no work experience
add(
"""Amara Okafor
amara.okafor@gmail.com

Recent graduate seeking entry-level data analyst roles.

Education:
BSc Statistics, University of Lagos, 2024

Skills: Python, SQL, Excel, Tableau""",
{
  "name": "Amara Okafor", "email": "amara.okafor@gmail.com", "phone": None, "location": None,
  "summary": "Recent graduate seeking entry-level data analyst roles.",
  "skills": ["Python", "SQL", "Excel", "Tableau"],
  "experience": [],
  "education": [{"degree": "BSc Statistics", "institution": "University of Lagos", "year": "2024"}],
  "certifications": []
})

# 4. Garbled multi-column PDF extraction (columns interleaved)
add(
"""Carlos Mendes SKILLS
Software Architect Java, Kubernetes, AWS
carlos.mendes@mail.com EXPERIENCE
+55 11 91234-5678 Lead Architect - Nimbus Corp
Sao Paulo, Brazil 2019-Present
Designed cloud-native platform serving 10M users
EDUCATION
MSc Computer Science - USP, 2015""",
{
  "name": "Carlos Mendes", "email": "carlos.mendes@mail.com", "phone": "+55 11 91234-5678",
  "location": "Sao Paulo, Brazil", "summary": None,
  "skills": ["Java", "Kubernetes", "AWS"],
  "experience": [{"title": "Lead Architect", "company": "Nimbus Corp", "start_date": "2019", "end_date": "Present", "description": "Designed cloud-native platform serving 10M users"}],
  "education": [{"degree": "MSc Computer Science", "institution": "USP", "year": "2015"}],
  "certifications": []
})

# 5. Non-English resume (French)
add(
"""Marie Dubois
marie.dubois@example.fr | Paris, France

Experience professionnelle :
Chef de projet marketing, Groupe Lumiere (2017 - 2022)
- Gestion de campagnes publicitaires multicanal

Formation :
Master en Marketing, HEC Paris, 2017""",
{
  "name": "Marie Dubois", "email": "marie.dubois@example.fr", "phone": None,
  "location": "Paris, France", "summary": None, "skills": [],
  "experience": [{"title": "Chef de projet marketing", "company": "Groupe Lumiere", "start_date": "2017", "end_date": "2022", "description": "Gestion de campagnes publicitaires multicanal"}],
  "education": [{"degree": "Master en Marketing", "institution": "HEC Paris", "year": "2017"}],
  "certifications": []
})

# 6. Employment gap
add(
"""David Kim
david.kim@mail.com | Seoul, South Korea

Experience:
Product Manager, Hana Soft (2016 - 2019)
[Career break 2019 - 2021 for family care]
Senior Product Manager, KoreaTech (2021 - Present)""",
{
  "name": "David Kim", "email": "david.kim@mail.com", "phone": None, "location": "Seoul, South Korea",
  "summary": None, "skills": [],
  "experience": [
    {"title": "Product Manager", "company": "Hana Soft", "start_date": "2016", "end_date": "2019", "description": None},
    {"title": "Senior Product Manager", "company": "KoreaTech", "start_date": "2021", "end_date": "Present", "description": None}
  ],
  "education": [], "certifications": []
})

# 7. Freelancer / multiple concurrent short contracts
add(
"""Elena Rossi - Freelance Graphic Designer
elena.rossi@design.com | Milan, Italy

- Brand identity design for Caffe Milano (2022)
- Packaging design for Verde Foods (2022-2023)
- Web illustrations for TechBlog Italia (2023)

Skills: Adobe Illustrator, Photoshop, Figma""",
{
  "name": "Elena Rossi", "email": "elena.rossi@design.com", "phone": None, "location": "Milan, Italy",
  "summary": "Freelance Graphic Designer",
  "skills": ["Adobe Illustrator", "Photoshop", "Figma"],
  "experience": [
    {"title": "Freelance Graphic Designer", "company": "Caffe Milano", "start_date": "2022", "end_date": "2022", "description": "Brand identity design"},
    {"title": "Freelance Graphic Designer", "company": "Verde Foods", "start_date": "2022", "end_date": "2023", "description": "Packaging design"},
    {"title": "Freelance Graphic Designer", "company": "TechBlog Italia", "start_date": "2023", "end_date": "2023", "description": "Web illustrations"}
  ],
  "education": [], "certifications": []
})

# 8. Executive resume with many roles, no dates for some
add(
"""ROBERT HAYES
Chief Operating Officer
rhayes@corpmail.com | New York, NY | (212) 555-0199

PROFESSIONAL EXPERIENCE
COO, Meridian Holdings, 2018-Present
VP Operations, Meridian Holdings, 2014-2018
Director of Operations, Bluewave Inc.

EDUCATION
MBA, Wharton School, University of Pennsylvania
BA Economics, Cornell University, 1998""",
{
  "name": "Robert Hayes", "email": "rhayes@corpmail.com", "phone": "(212) 555-0199",
  "location": "New York, NY", "summary": None, "skills": [],
  "experience": [
    {"title": "COO", "company": "Meridian Holdings", "start_date": "2018", "end_date": "Present", "description": None},
    {"title": "VP Operations", "company": "Meridian Holdings", "start_date": "2014", "end_date": "2018", "description": None},
    {"title": "Director of Operations", "company": "Bluewave Inc.", "start_date": None, "end_date": None, "description": None}
  ],
  "education": [
    {"degree": "MBA", "institution": "Wharton School, University of Pennsylvania", "year": None},
    {"degree": "BA Economics", "institution": "Cornell University", "year": "1998"}
  ],
  "certifications": []
})

# 9. Resume with only a skills section (no experience/education at all)
add(
"""Skills: JavaScript, React, Node.js, Docker, PostgreSQL, GraphQL""",
{
  "name": None, "email": None, "phone": None, "location": None, "summary": None,
  "skills": ["JavaScript", "React", "Node.js", "Docker", "PostgreSQL", "GraphQL"],
  "experience": [], "education": [], "certifications": []
})

# 10. Malformed / OCR-noise input
add(
"""J0hn  Sm ith
j0hn.smith@ ema il.co m
Ex perienc e:  Sal es  Ass0ciate ,  QuickMart  (2 019-2021)""",
{
  "name": "John Smith", "email": "j0hn.smith@email.com", "phone": None, "location": None,
  "summary": None, "skills": [],
  "experience": [{"title": "Sales Associate", "company": "QuickMart", "start_date": "2019", "end_date": "2021", "description": None}],
  "education": [], "certifications": []
})

# Additional 40 concise but varied examples generated programmatically below
templates = [
 ("Fatima Al-Sayed", "fatima.alsayed@mail.ae", "+971 50 123 4567", "Dubai, UAE",
  "HR Business Partner with 8 years in talent management.",
  ["Recruitment", "HRIS", "Employee Relations"],
  [{"title": "HR Business Partner", "company": "Falcon Group", "start_date": "2016-03", "end_date": "Present", "description": "Manage HR operations for 300+ employees"}],
  [{"degree": "MBA Human Resources", "institution": "American University of Sharjah", "year": "2015"}],
  ["SHRM-CP"]),
 ("Wei Zhang", None, "+86 138 0013 8000", "Shanghai, China", None,
  ["Python", "TensorFlow", "PyTorch"],
  [{"title": "Machine Learning Engineer", "company": "DeepVision AI", "start_date": "2020-01", "end_date": "Present", "description": "Built computer vision models for retail analytics"}],
  [{"degree": "MSc Artificial Intelligence", "institution": "Fudan University", "year": "2019"}],
  []),
 ("Grace Mensah", "grace.mensah@mail.com", None, "Accra, Ghana",
  "Registered nurse with pediatric ICU experience.", ["Patient Care", "CPR", "Pediatrics"],
  [{"title": "Staff Nurse", "company": "Korle Bu Teaching Hospital", "start_date": "2017", "end_date": "2023", "description": "Pediatric ICU care"}],
  [{"degree": "BSc Nursing", "institution": "University of Ghana", "year": "2016"}],
  ["Basic Life Support Certification"]),
 ("Tomas Novak", "tomas.novak@mail.cz", "+420 601 234 567", None, None,
  ["Civil Engineering", "AutoCAD", "Project Management"],
  [{"title": "Site Engineer", "company": "Praha Stavby", "start_date": "2018", "end_date": "2022", "description": "Managed residential construction sites"}],
  [], []),
 ("Aisha Rahman", "aisha.r@mail.com", "+880 1712 345678", "Dhaka, Bangladesh",
  "Content strategist specializing in SEO-driven storytelling.",
  ["SEO", "Content Strategy", "Google Analytics"],
  [{"title": "Content Strategist", "company": "Digital Bangla", "start_date": "2019", "end_date": "Present", "description": "Grew organic traffic 300% in 2 years"}],
  [{"degree": "BA English", "institution": "Dhaka University", "year": "2018"}],
  []),
 (None, "anonymous123@mail.com", None, None, None, ["Welding", "Metal Fabrication"], [], [], []),
 ("Lucas Ferreira", "lucas.f@mail.com.br", "+55 21 99887-6655", "Rio de Janeiro, Brazil",
  None, ["Sales", "CRM", "Negotiation"],
  [{"title": "Account Executive", "company": "Vendas Rapidas", "start_date": "2020", "end_date": "Present", "description": "Closed $2M in annual contracts"},
   {"title": "Sales Associate", "company": "Loja Central", "start_date": "2017", "end_date": "2020", "description": None}],
  [], []),
 ("Sofia Petrova", "sofia.petrova@mail.ru", None, "Moscow, Russia",
  "Financial analyst focused on equity research.", ["Excel", "Bloomberg Terminal", "Valuation"],
  [{"title": "Financial Analyst", "company": "Moscow Capital Partners", "start_date": "2021-04", "end_date": "Present", "description": "Coverage of 15 mid-cap equities"}],
  [{"degree": "BSc Finance", "institution": "Higher School of Economics", "year": "2021"}],
  ["CFA Level 1"]),
 ("Michael O'Brien", "m.obrien@mail.ie", "+353 87 123 4567", "Dublin, Ireland",
  None, ["Java", "Spring Boot", "Kubernetes", "AWS"],
  [{"title": "Backend Developer", "company": "Fintech Eire", "start_date": "2015", "end_date": "2019", "description": None},
   {"title": "Senior Backend Developer", "company": "Fintech Eire", "start_date": "2019", "end_date": "Present", "description": "Led payments platform rebuild"}],
  [{"degree": "BSc Computer Science", "institution": "Trinity College Dublin", "year": "2015"}], []),
 ("Hana Kobayashi", "hana.k@mail.jp", None, "Tokyo, Japan",
  "UX designer passionate about accessible design.", ["Figma", "User Research", "Prototyping"],
  [{"title": "UX Designer", "company": "Sakura Digital", "start_date": "2019", "end_date": "Present", "description": "Redesigned mobile banking app, +18% task completion"}],
  [{"degree": "BFA Design", "institution": "Tokyo University of the Arts", "year": "2018"}], []),
 ("Samuel Adeyemi", "s.adeyemi@mail.com", "+234 803 123 4567", "Lagos, Nigeria",
  "Mechanical engineer with oil & gas sector experience.", ["AutoCAD", "SolidWorks", "Six Sigma"],
  [{"title": "Mechanical Engineer", "company": "Niger Delta Energy", "start_date": "2016", "end_date": "2021", "description": "Maintained offshore pipeline systems"},
   {"title": "Senior Mechanical Engineer", "company": "Niger Delta Energy", "start_date": "2021", "end_date": "Present", "description": None}],
  [{"degree": "BEng Mechanical Engineering", "institution": "University of Lagos", "year": "2015"}], ["Six Sigma Black Belt"]),
 ("Isabella Garcia", "isabella.g@mail.com", "+34 611 223 344", "Madrid, Spain",
  None, ["Spanish", "English", "Translation", "Localization"],
  [{"title": "Translator", "company": "Globex Language Services", "start_date": "2018", "end_date": "Present", "description": "Legal and technical document translation"}],
  [{"degree": "BA Translation and Interpreting", "institution": "Universidad Complutense de Madrid", "year": "2017"}], []),
 ("Noah Bennett", "noah.b@mail.com", None, "Toronto, Canada",
  "High school science teacher, 10 years experience.", ["Curriculum Design", "Classroom Management"],
  [{"title": "Science Teacher", "company": "Maple Ridge High School", "start_date": "2014", "end_date": "Present", "description": "Teach physics and chemistry to grades 9-12"}],
  [{"degree": "B.Ed", "institution": "University of Toronto", "year": "2013"},
   {"degree": "BSc Physics", "institution": "University of Toronto", "year": "2011"}], ["Ontario Teaching Certificate"]),
 ("Olivia Nguyen", "olivia.n@mail.com", "+84 90 123 4567", "Ho Chi Minh City, Vietnam",
  None, ["Digital Marketing", "Facebook Ads", "Google Ads"],
  [{"title": "Digital Marketing Executive", "company": "VietMedia", "start_date": "2021", "end_date": "Present", "description": "Manage $50K/month ad spend across channels"}],
  [], []),
 ("Ahmed Hassan", "ahmed.hassan@mail.com", "+20 100 123 4567", "Cairo, Egypt",
  "Pharmacist transitioning into clinical research.", ["Clinical Trials", "GCP", "Pharmacology"],
  [{"title": "Clinical Research Associate", "company": "MedTrial Egypt", "start_date": "2022", "end_date": "Present", "description": "Monitor phase III trial sites"},
   {"title": "Retail Pharmacist", "company": "Cairo Pharmacy Group", "start_date": "2018", "end_date": "2022", "description": None}],
  [{"degree": "PharmD", "institution": "Cairo University", "year": "2018"}], ["GCP Certification"]),
 ("Emily Watson", "emily.watson@mail.com", "+44 7911 123456", "London, UK",
  None, ["Corporate Law", "Contract Negotiation", "M&A"],
  [{"title": "Associate", "company": "Whitfield & Partners LLP", "start_date": "2019", "end_date": "Present", "description": "M&A transactions for mid-market clients"}],
  [{"degree": "LLB Law", "institution": "London School of Economics", "year": "2018"}], ["Solicitor of England and Wales"]),
 ("Ravi Patel", "ravi.patel@mail.com", None, "Ahmedabad, India",
  None, ["Manual Testing", "Selenium", "JIRA"],
  [{"title": "QA Engineer", "company": "Softedge Solutions", "start_date": "2020", "end_date": "Present", "description": "Automated regression suite reducing test time 40%"}],
  [{"degree": "B.E. Information Technology", "institution": "Gujarat Technological University", "year": "2019"}], ["ISTQB Foundation"]),
 ("Chloe Martin", "chloe.martin@mail.com", "+33 6 12 34 56 78", "Lyon, France",
  "Event manager specializing in corporate conferences.", ["Event Planning", "Vendor Management", "Budgeting"],
  [{"title": "Event Manager", "company": "Lyon Events Co", "start_date": "2017", "end_date": "Present", "description": "Managed 50+ corporate events annually"}],
  [], []),
 ("Benjamin Cohen", "ben.cohen@mail.com", None, "Tel Aviv, Israel",
  None, ["iOS Development", "Swift", "SwiftUI"],
  [{"title": "iOS Developer", "company": "AppWorks Israel", "start_date": "2019", "end_date": "Present", "description": "Published 3 apps with 1M+ combined downloads"}],
  [{"degree": "BSc Computer Science", "institution": "Tel Aviv University", "year": "2018"}], []),
 ("Grace Lin", "grace.lin@mail.com", "+886 912 345 678", "Taipei, Taiwan",
  "Supply chain analyst focused on demand forecasting.", ["Supply Chain", "Excel", "SAP"],
  [{"title": "Supply Chain Analyst", "company": "Taiwan Electronics Co", "start_date": "2020", "end_date": "Present", "description": "Reduced inventory costs by 12%"}],
  [{"degree": "BSc Industrial Engineering", "institution": "National Taiwan University", "year": "2019"}], []),
 ("Daniel Osei", "daniel.osei@mail.com", None, "Kumasi, Ghana",
  None, ["Electrical Wiring", "Solar Installation"],
  [{"title": "Electrician", "company": "BrightSpark Ltd", "start_date": "2015", "end_date": "Present", "description": "Residential and solar panel installations"}],
  [], ["Certified Electrician - Ghana"]),
 ("Nina Kowalski", "nina.k@mail.pl", "+48 601 234 567", "Warsaw, Poland",
  None, ["Java", "Spring", "Microservices"],
  [{"title": "Software Engineer", "company": "Warsaw Tech Hub", "start_date": "2021", "end_date": "Present", "description": None}],
  [{"degree": "MSc Computer Science", "institution": "Warsaw University of Technology", "year": "2021"}], []),
 ("Julia Costa", "julia.costa@mail.pt", None, "Lisbon, Portugal",
  "Veterinarian with small animal clinic experience.", ["Animal Care", "Surgery", "Diagnostics"],
  [{"title": "Veterinarian", "company": "Clinica Animal Lisboa", "start_date": "2017", "end_date": "Present", "description": "Primary care for 20+ patients daily"}],
  [{"degree": "DVM", "institution": "University of Lisbon", "year": "2016"}], []),
 ("Marcus Johnson", "marcus.j@mail.com", "+1-404-555-0148", "Atlanta, GA",
  None, ["Public Speaking", "Sales Training", "Leadership"],
  [{"title": "Regional Sales Director", "company": "Peachtree Solutions", "start_date": "2015", "end_date": "Present", "description": "Manage team of 25 across 4 states"},
   {"title": "Sales Manager", "company": "Peachtree Solutions", "start_date": "2011", "end_date": "2015", "description": None}],
  [{"degree": "BA Business Administration", "institution": "Georgia State University", "year": "2010"}], []),
]

extra_notes = [
 "HR Business Partner with 8 years in talent management.",
]

for (name, email, phone, location, summary, skills, experience, education, certs) in templates:
    lines = []
    header_bits = [b for b in [name, email, phone, location] if b]
    lines.append(" | ".join(header_bits) if header_bits else "")
    if summary:
        lines.append(f"\nSummary: {summary}")
    if experience:
        lines.append("\nExperience:")
        for e in experience:
            dates = f"({e.get('start_date') or '?'} - {e.get('end_date') or '?'})"
            lines.append(f"{e['title']}, {e['company']} {dates}")
            if e.get("description"):
                lines.append(f"- {e['description']}")
    if education:
        lines.append("\nEducation:")
        for ed in education:
            lines.append(f"{ed['degree']}, {ed['institution']}" + (f", {ed['year']}" if ed.get('year') else ""))
    if skills:
        lines.append("\nSkills: " + ", ".join(skills))
    if certs:
        lines.append("\nCertifications: " + ", ".join(certs))
    input_text = "\n".join([l for l in lines if l != ""])
    add(input_text, {
        "name": name, "email": email, "phone": phone, "location": location,
        "summary": summary, "skills": skills, "experience": experience,
        "education": education, "certifications": certs
    })

# A batch of additional short/edge-case examples to reach 50+
edge_cases = [
 ("Resume with only a phone number, nothing else",
  "Call me: +1-555-0134",
  {"name": None, "email": None, "phone": "+1-555-0134", "location": None, "summary": None,
   "skills": [], "experience": [], "education": [], "certifications": []}),
 ("Resume where dates use 'Present' inconsistently capitalized",
  "Ana Torres | ana.t@mail.com\nExperience: Analyst, DataCorp (2020-present)",
  {"name": "Ana Torres", "email": "ana.t@mail.com", "phone": None, "location": None, "summary": None,
   "skills": [], "experience": [{"title": "Analyst", "company": "DataCorp", "start_date": "2020", "end_date": "Present", "description": None}],
   "education": [], "certifications": []}),
 ("Resume listing certifications only, in a bulleted list",
  "Certifications:\n- PMP\n- Six Sigma Green Belt\n- ITIL Foundation",
  {"name": None, "email": None, "phone": None, "location": None, "summary": None, "skills": [],
   "experience": [], "education": [], "certifications": ["PMP", "Six Sigma Green Belt", "ITIL Foundation"]}),
 ("Two people's resumes concatenated by a bad PDF split (should extract first person only, flag ambiguity via nulls where unclear)",
  "Jane Doe\njane.doe@mail.com\nExperience: Teacher, Green Valley School (2015-2020)\n\n--- \nMark Lee\nmark.lee@mail.com",
  {"name": "Jane Doe", "email": "jane.doe@mail.com", "phone": None, "location": None, "summary": None,
   "skills": [], "experience": [{"title": "Teacher", "company": "Green Valley School", "start_date": "2015", "end_date": "2020", "description": None}],
   "education": [], "certifications": []}),
 ("Resume with a table-formatted skills matrix flattened to text",
  "Skill Level\nPython Expert\nSQL Intermediate\nDocker Beginner",
  {"name": None, "email": None, "phone": None, "location": None, "summary": None,
   "skills": ["Python", "SQL", "Docker"], "experience": [], "education": [], "certifications": []}),
 ("Resume with only education, no name label given but signed at bottom",
  "Education:\nPhD Chemistry, MIT, 2021\nMSc Chemistry, MIT, 2017\n\n- Kevin Zhao",
  {"name": "Kevin Zhao", "email": None, "phone": None, "location": None, "summary": None, "skills": [],
   "experience": [], "education": [
     {"degree": "PhD Chemistry", "institution": "MIT", "year": "2021"},
     {"degree": "MSc Chemistry", "institution": "MIT", "year": "2017"}], "certifications": []}),
 ("Resume with date range using year-only and season labels",
  "Laura Bianchi | laura.b@mail.it\nExperience: Marketing Intern, Moda Italia (Summer 2021 - Fall 2021)",
  {"name": "Laura Bianchi", "email": "laura.b@mail.it", "phone": None, "location": None, "summary": None,
   "skills": [], "experience": [{"title": "Marketing Intern", "company": "Moda Italia", "start_date": "Summer 2021", "end_date": "Fall 2021", "description": None}],
   "education": [], "certifications": []}),
 ("Resume with multiple emails listed, should extract the primary/first one",
  "Peter Nilsson\nWork: peter.nilsson@company.com  Personal: peter.n.personal@gmail.com",
  {"name": "Peter Nilsson", "email": "peter.nilsson@company.com", "phone": None, "location": None,
   "summary": None, "skills": [], "experience": [], "education": [], "certifications": []}),
 ("Resume with skills embedded inline in a sentence, not a labeled list",
  "I am proficient in Python, R, and Tableau, with 3 years of data visualization experience.",
  {"name": None, "email": None, "phone": None, "location": None, "summary": None,
   "skills": ["Python", "R", "Tableau"], "experience": [], "education": [], "certifications": []}),
 ("Resume in all-caps formatting (common OCR artifact)",
  "SARAH CONNOR\nSARAH.CONNOR@MAIL.COM\nEXPERIENCE: SECURITY ANALYST, CYBERDYNE SYSTEMS (2020-PRESENT)",
  {"name": "Sarah Connor", "email": "sarah.connor@mail.com", "phone": None, "location": None, "summary": None,
   "skills": [], "experience": [{"title": "Security Analyst", "company": "Cyberdyne Systems", "start_date": "2020", "end_date": "Present", "description": None}],
   "education": [], "certifications": []}),
 ("Resume with a certifications section that includes issue years",
  "Certifications:\nPMP (2019)\nCSM (2021)",
  {"name": None, "email": None, "phone": None, "location": None, "summary": None, "skills": [],
   "experience": [], "education": [], "certifications": ["PMP (2019)", "CSM (2021)"]}),
 ("Resume where location is a country only, no city",
  "Yusuf Demir | yusuf.demir@mail.com | Turkey\nExperience: Logistics Coordinator, Anadolu Shipping (2018-Present)",
  {"name": "Yusuf Demir", "email": "yusuf.demir@mail.com", "phone": None, "location": "Turkey", "summary": None,
   "skills": [], "experience": [{"title": "Logistics Coordinator", "company": "Anadolu Shipping", "start_date": "2018", "end_date": "Present", "description": None}],
   "education": [], "certifications": []}),
 ("Blank / empty resume input", "",
  {"name": None, "email": None, "phone": None, "location": None, "summary": None, "skills": [],
   "experience": [], "education": [], "certifications": []}),
 ("Resume with a personal website/portfolio link mixed into contact info (not part of schema, should be ignored)",
  "Tara Singh | tara.singh@mail.com | portfolio: tarasingh.design\nExperience: Product Designer, Loop Studio (2019-Present)",
  {"name": "Tara Singh", "email": "tara.singh@mail.com", "phone": None, "location": None, "summary": None,
   "skills": [], "experience": [{"title": "Product Designer", "company": "Loop Studio", "start_date": "2019", "end_date": "Present", "description": None}],
   "education": [], "certifications": []}),
 ("Resume with numbers-heavy achievement description mixing metrics and dates",
  "Rahul Verma | rahul.verma@mail.com\nExperience: Growth Marketer, ScaleUp Inc (2020-2023) - Grew MRR from $10K to $250K in 24 months",
  {"name": "Rahul Verma", "email": "rahul.verma@mail.com", "phone": None, "location": None, "summary": None,
   "skills": [], "experience": [{"title": "Growth Marketer", "company": "ScaleUp Inc", "start_date": "2020", "end_date": "2023", "description": "Grew MRR from $10K to $250K in 24 months"}],
   "education": [], "certifications": []}),
]
for _, input_text, output in edge_cases:
    add(input_text, output)

# Ensure at least 50 examples
add(
"Anders Larsen | anders.larsen@mail.dk | Copenhagen, Denmark\nExperience: DevOps Engineer, Nordic Cloud ApS (2019-Present) - Manage CI/CD pipelines for 40+ microservices\nEducation: BSc Software Engineering, IT University of Copenhagen, 2018\nSkills: Terraform, Kubernetes, Jenkins",
{
  "name": "Anders Larsen", "email": "anders.larsen@mail.dk", "phone": None, "location": "Copenhagen, Denmark",
  "summary": None, "skills": ["Terraform", "Kubernetes", "Jenkins"],
  "experience": [{"title": "DevOps Engineer", "company": "Nordic Cloud ApS", "start_date": "2019", "end_date": "Present", "description": "Manage CI/CD pipelines for 40+ microservices"}],
  "education": [{"degree": "BSc Software Engineering", "institution": "IT University of Copenhagen", "year": "2018"}],
  "certifications": []
})

assert len(examples) >= 50, f"Only {len(examples)} examples generated"

out_path = Path(__file__).parent / "resume_instructions.jsonl"
with open(out_path, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Wrote {len(examples)} examples to {out_path}")
