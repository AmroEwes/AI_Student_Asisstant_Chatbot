import os, time, ast, pandas as pd, streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from gpt4all import GPT4All

# ───────────── 0.  MODEL BACKEND TOGGLE ─────────────
MODEL_OPTIONS = {
    "OpenAI GPT-4": "openai",
    "Mistral 7B (mistral-7b-instruct)": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "Nous Hermes 2 (Mistral 7B)": "nous-hermes-2-mistral-7b-dpo.Q4_K_M.gguf",
    "MythoMax L2 (LLaMA 13B)": "mythomax-l2-13b.Q4_K_M.gguf",
    "OpenHermes 2.5 (Mistral 7B)": "openhermes-2.5-mistral-7b.Q4_K_M.gguf"
}
MODEL_LABELS = list(MODEL_OPTIONS.keys())
SELECTED_MODEL_LABEL = st.sidebar.selectbox("Model Backend", MODEL_LABELS, index=0)
SELECTED_MODEL = MODEL_OPTIONS[SELECTED_MODEL_LABEL]

# ───────────── 1.  CONFIG + CSS ─────────────
with st.sidebar:
    st.image("chatbot.png", width=180)  # Adjust width as needed
    st.markdown("### GPT Powered")
    st.markdown("---")

st.set_page_config("🎓 AI Student Asisstant", layout="centered")
st.title("🎓 AI Student Asisstant")
# Add custom CSS to the Streamlit app
st.markdown(
    """
    <style>
    /* Ensure the font-family applies to all text elements */
    @font-face {
        font-family: 'Times New Roman';
        src: url('https://fonts.cdnfonts.com/s/15292/Times_New_Roman.woff') format('woff');
    }
    body, div, p, h1, h2, h3, h4, h5, h6, span, td, th, li, label, input, button, select, textarea, .stMarkdown, .stTextInput, .stTextArea, .stRadio, .stCheckbox, .stSelectbox, .stMultiSelect, .stButton, .stSlider, .stDataFrame, .stTable, .stExpander, .stTabs, .stAccordion, .stDownloadButton {
        font-family: 'Times New Roman', serif !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("""
<style>
/* Applies to all Streamlit buttons */
.stButton > button {
    font-size: 14px !important;
    padding: 0.9em 1em !important;
    white-space: normal !important;   /* allow wrapping */
    word-wrap: break-word !important; /* long words wrap */
    text-align: center !important;    /* center multiline text */
}
</style>
""", unsafe_allow_html=True)


# ───────────── 2.  LOAD FILES ─────────────
df_major  = pd.read_csv("docs/MajorSheet.csv")
df_major["Course_ID"] = df_major["Course_ID"].astype(str).str.strip().str.upper()
req_df    = pd.read_csv("docs/Major_Sheet_Requirements.csv")

master_df   = pd.read_csv("docs/Students_Master_Data.csv", dtype={"StudentID": str})
enroll_df   = pd.read_csv("docs/Students_Enrollment_Data.csv", dtype={"StudentID": str})
progress_df = pd.read_csv("docs/student_progress.csv", dtype={"Student_ID": str})
elig_list   = pd.read_csv("docs/Eligible_Course_List.csv", dtype={"Student_ID": str})
elig_det    = pd.read_csv("docs/Eligible_Course_Details.csv", dtype={"Student_ID": str})
schedule_df = pd.read_csv("docs/Current_Schedule_Data.csv")
schedule_df["CourseID"] = schedule_df["CourseID"].astype(str).str.strip().str.upper()
current_semester = schedule_df["Semester"].max()   # or hard-code, e.g. 2403


progress_df["Student_Progress"] = progress_df["Student_Progress"].round(2).astype(str) + "%"

# ───────────── 3.  MAPPINGS ─────────────
major_map = {
    "ACCOUNTING":"Accounting","INTL BUSIN":"International Business","MANAGEMENT":"Management",
    "COMSCIENCE":"Computer Science","COMPENG":"Computer Engineering","ELECENG":"Electrical Engineering",
    "MGMTENG":"Engineering Management","ENGLISH":"English Education","LINGUISTIC":"Linguistics & Translation",
    "LITERATURE":"English Literature","FINANCE":"Finance","DIGITALMED":"Digital Media Production",
    "PR / ADV":"Public Relations & Advertising","VISUAL COM":"Visual Communication",
    "MIS":"Management Information Systems","MARKETING2":"Marketing"}
rev_map = {v:k for k,v in major_map.items()}

area_map = {
        "CR": "🟩 Core Requirement",
        "GE": "🟦 General Education",
        "ME": "🟨 Major Elective",
        "MR": "🟥 Major Requirement",
        "EB": "🟪 Engineering Breadth",
        "NA": "⬜ Not Applicable",
        "EFU": "🟫 English Foundation Unit",
        "MFU": "⬛ Math Foundation Unit",
        "FE": "🟧 Free Elective"  # NEW TAG
    }

# ───────────── 4.  MODEL LOADING ─────────────
if SELECTED_MODEL == "openai":
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
else:
    MODEL_PATH = os.path.join(os.getcwd(), "models", SELECTED_MODEL)
    @st.cache_resource(show_spinner=f"Loading {SELECTED_MODEL_LABEL} model...")
    def load_gpt4all_model(model_path):
        return GPT4All(model_path)
    gpt4all_model = load_gpt4all_model(MODEL_PATH)

def local_gpt4all(prompt, max_tokens=512):
    with gpt4all_model.chat_session():
        response = gpt4all_model.generate(
            prompt,
            max_tokens=max_tokens,
            temp=0.7,
            top_p=0.95,
            n_batch=8,
            streaming=False
        )
    return response.strip()

# ───────────── 4.  OPENAI/GPT4ALL GPT FUNCTIONS ─────────────
def gpt_explain_course(row, major_name):
    advisor_prompt = (
        "You are an academic advisor assistant. You use structured course data to explain how a course applies to a student's major. \n\n"
        "📚 AREA_OF_STUDY:\n"
        "- CR = College Requirement\n"
        "- GE = General Education\n"
        "- ME = Major Elective\n"
        "- MR = Major Requirement\n"
        "- NA = Not applicable (Free Elective)\n"
        "- EB = Engineering Breadth\n"
        "- EFU = English Foundation Unit\n"
        "- MFU = Math Foundation Unit\n\n"
        "📚 COURSE_OF_STUDY:\n"
        "- R = Required\n"
        "- RE = Required Elective\n"
        "- E = Elective\n"
        "- N = Not applicable\n"
        "- X = Requires department consent\n"
        "- Z = Not included\n\n"
        "Use this info to explain the course role in the major, eligibility logic, and prerequisites."
    )
    row_txt = "\n".join([f"{k}: {v}" for k, v in row.items()])
    sys = "You are an academic-advisor assistant. Explain clearly if the course is required, elective, or not applicable and list any prerequisites, given AREA_OF_STUDY & COURSE_OF_STUDY codes."
    usr = f"I am a {major_name} student.\n{row_txt}"
    # Prepend advisor_prompt to system prompt
    full_sys = advisor_prompt + "\n\n" + sys
    prompt = f"System: {full_sys}\nUser: {usr}\nAssistant:"
    if SELECTED_MODEL == "openai":
        resp = client.chat.completions.create(model="gpt-4-turbo",
            messages=[{"role":"system","content":full_sys},{"role":"user","content":usr}])
        return resp.choices[0].message.content
    else:
        return local_gpt4all(prompt)

# ───────────── 5.  HELPER FUNCTIONS ─────────────
def student_profile_text(sid):
    match = master_df[master_df["StudentID"] == sid]
    if match.empty:
        return f"❌ No student information found for ID `{sid}`."
    r = match.iloc[0]
    return (f"🧑 **Student ID:** {r.StudentID}\n"
            f"- 👤 - **Name:** {r.FullName}\n"
            f"- 🎓 - **Major:** {r.Major}\n"
            f"- 🏫 - **College:** {r.CollegeDescription}\n"
            f"- 📚 - **Department:** {r.DepartmentDescription}\n"
            f"- 📅 - **Admit Semester:** {r.AdmitSemester}\n"
            f"- 👨‍🏫 - **Advisor Name:** {r.AdvisorFullName}\n"
            f"- 📧 - **Advisor Email:** {r.AdvisorEmail}\n")

def enrolled_text(sid):
    df=enroll_df[enroll_df["StudentID"]==sid]
    if df.empty: return "📚 No courses taken yet."
    lines=[f"- {c}: Grade {g}" for c,g in
           df.groupby("Course_ID")["LetterGrade"].last().reset_index().values]
    return "📚 **Courses Taken:**\n"+"\n".join(lines)

def student_progress_text(sid):
    match = progress_df[progress_df["Student_ID"] == sid]
    if match.empty:
        return f"📊 No progress record found for `{sid}`."

    summary = f"📊 **Progress by Area of Study for Student {sid}:**\n\n"
    for _, row in match.iterrows():
        area = row["AREA_OF_STUDY"]
        area_full = area_map.get(area, area)
        
        taken_courses = int(row["Total_Taken_Courses"])
        required_courses = int(row["Required_Courses"])
        remaining_courses = int(row["Remaining_Courses"])
        progress = row["Student_Progress"]

        summary += (
            f"• **{area}** ({area_full}) – "
            f"{taken_courses}/{required_courses} course(s) completed.\n"
            f"  Remaining: {remaining_courses} course(s).\n"
            f"  Progress: {progress}\n\n"
        )
    return summary


def eligible_text(sid, completed):
    df=elig_det[(elig_det["Student_ID"]==sid) & (~elig_det["AREA_OF_STUDY"].isin(completed))]
    if df.empty: return "✅ No eligible courses remaining in incomplete areas."
    out="✅ **Eligible Courses (areas not yet complete):**\n\n"
    for area,grp in df.groupby("AREA_OF_STUDY"):
        out+=f"**{area}** ({area_map.get(area,area)}):\n"
        out+="- "+" \n- ".join(grp["Eligible_Courses"].tolist())+"\n\n"
    return out

def offered_this_term(sid):
    """Return a list of eligible CourseIDs that are actually offered this semester."""
    # 1. fetch eligible list
    elig_row = elig_list[elig_list["Student_ID"] == sid]
    if elig_row.empty: return []
    elig_raw = elig_row.iloc[0]["Eligible_Courses_List"]
    elig_codes = ast.literal_eval(elig_raw) if isinstance(elig_raw, str) else []
    elig_codes = [c.strip().upper() for c in elig_codes]

    # 2. courses the student already took
    taken = enroll_df[enroll_df["StudentID"] == sid]["Course_ID"].str.upper().tolist()

    # 3. offered this semester
    offered_now = schedule_df[schedule_df["Semester"] == current_semester]["CourseID"].unique().tolist()

    # 4. keep only eligible, offered, and not previously taken
    final = [c for c in elig_codes if (c in offered_now and c not in taken)]
    return final

def gpt_recommend_courses(sid, major_name, summary_text, offered_list):
    """
    Recommends up to 5 eligible and offered courses using GPT, based on student progress and course info.
    """

    if not offered_list:
        return "❌ No eligible courses are offered this semester."

    # Join with eligibility details
    offered_df = elig_det[
        (elig_det["Student_ID"] == sid) & 
        (elig_det["Eligible_Courses"].isin(offered_list))
    ][["Eligible_Courses", "AREA_OF_STUDY", "COURSE_OF_STUDY"]].drop_duplicates()

    # Add CourseDescription, CourseCode, and CourseLongDescription
    course_info_df = schedule_df[
        schedule_df["CourseID"].isin(offered_list)
    ][["CourseID", "CourseDescription", "CourseCode", "CourseLongDescription"]].drop_duplicates()

    course_desc_map = course_info_df.set_index("CourseID")["CourseDescription"].to_dict()
    course_long_map = course_info_df.set_index("CourseID")["CourseLongDescription"].to_dict()
    course_code_map = course_info_df.set_index("CourseID")["CourseCode"].to_dict()

    # Merge details
    offered_df["CourseDescription"] = offered_df["Eligible_Courses"].map(course_desc_map)
    offered_df["CourseLongDescription"] = offered_df["Eligible_Courses"].map(course_long_map)
    offered_df["CourseCode"] = offered_df["Eligible_Courses"].map(course_code_map)

    # Convert CourseCode to int for sorting
    def safe_code(val):
        try:
            return int(str(val).strip())
        except:
            return 9999
    offered_df["CourseCode"] = offered_df["CourseCode"].apply(safe_code)

    # Sort by CourseCode ascending
    offered_df = offered_df.sort_values("CourseCode")

    # Compose the eligible/offered block
    offered_block = ""
    for _, row in offered_df.iterrows():
        cid = row["Eligible_Courses"]
        area = row["AREA_OF_STUDY"]
        cstudy = row["COURSE_OF_STUDY"]
        ccode = row["CourseCode"]
        title = row["CourseDescription"] or "No title."
        desc = row["CourseLongDescription"] or "No description."
        offered_block += f"- {cid} | Area: {area} | Type: {cstudy} | Code: {ccode}\n  • Title: {title}\n  • Description: {desc}\n\n"

    # Compose GPT prompt
    user_prompt = f"""
You are a university academic-advisor assistant (GPT-4).
Below is the student's academic profile, followed by eligible and offered courses.

================ STUDENT SUMMARY ================
{summary_text}

================ ELIGIBLE & OFFERED COURSES ================
{offered_block}

RECOMMENDATION RULES:
1. **First priority** → Recommend General Education (GE) courses with COURSE_OF_STUDY 'R' or 'RE'.
2. If fewer than 5 such GE courses exist, fill remaining slots with other unmet-area courses (prefer MR, CR, EB, then ME/GE/etc.)
3. Within any category, **Prefer lower `CourseCode` values** to suggest earlier, more suitable courses first.
4. NEVER suggest a course in an AREA_OF_STUDY the student has fully completed.
5. DO NOT select multiple courses from the same AreaOfStudy — aim to diversify the recommendations across different academic areas.
6. Use the **CourseDescription** and **CourseLongDescription** to justify why this course would help the student.
7. Return **up to five** bullet recommendations, in the format:

• **ENGL110 - English Composition I** – justification...
• **GEOL101 - Earth Science** – justification...
"""
    if SELECTED_MODEL == "openai":
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful academic advisor."},
                {"role": "user", "content": user_prompt}
            ]
        )
        return response.choices[0].message.content
    else:
        prompt = f"System: You are a helpful academic advisor.\nUser: {user_prompt}\nAssistant:"
        return local_gpt4all(prompt)

# ───────────── 6.  CHAT-STYLE UI ─────────────
# === Greeting Message (only once using session_state) ===
# === STEP 1: Chat UI with Memory ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.stage = "root"
    st.session_state.greet_done = False
    st.session_state.last_choice = None

# Helper to append assistant messages
def say_animated(text):
    st.session_state.chat_history.append({"role": "assistant", "content": text})
    container = st.empty()
    buf = ""

    # Build bubble as it types
    for ch in text:
        buf += ch

        # Convert bold markdown and line breaks
        temp_content = buf.replace("**", "<strong>").replace("\n", "<br>")
        while "**" in temp_content:
            temp_content = temp_content.replace("**", "<strong>", 1)
            temp_content = temp_content.replace("**", "</strong>", 1)

        # Re-render each frame with styled HTML
        container.markdown(
            f"""
            <div style='display: flex; flex-direction: row; align-items: flex-end; margin-bottom: 10px;'>
                <div style='font-size: 20px; margin-right: 8px;'>🤖</div>
                <div style='background: #F1F0F0; padding: 10px 14px; width: fit-content;
                            border-radius: 0 18px 18px 18px; font-family: "Segoe UI", sans-serif;
                            box-shadow: 0 1px 2px rgba(0,0,0,0.1); margin-right: auto;'>
                    {temp_content}▌
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        time.sleep(0.002)

    # Final render (remove cursor)
    final_content = text.replace("**", "<strong>").replace("\n", "<br>")
    while "**" in final_content:
        final_content = final_content.replace("**", "<strong>", 1)
        final_content = final_content.replace("**", "</strong>", 1)

    container.markdown(
        f"""
        <div style='display: flex; flex-direction: row; align-items: flex-end; margin-bottom: 10px;'>
            <div style='font-size: 20px; margin-right: 8px;'>🤖</div>
            <div style='background: #F1F0F0; padding: 10px 14px; width: fit-content;
                        border-radius: 0 18px 18px 18px; font-family: "Segoe UI", sans-serif;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.1); margin-right: auto;'>
                {final_content}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def show_selectbox(label, options, key):
    selected = st.selectbox(label, ["-- Select --"] + options, key=key)
    confirm = st.button("✅ Confirm Selection")
    return selected if confirm and selected != "-- Select --" else None

def render_chat_message(role, content):
    if role == "user":
        bubble_color = "#DCF8C6"  # light green
        border = "border-radius: 18px 0 18px 18px;"
        avatar = "🧑"
        flex_direction = "row-reverse"
        margin_style = "margin-left: auto;"  # Push to right
        avatar_margin = "margin-left: 8px;"
    else:
        bubble_color = "#F1F0F0"  # light gray
        border = "border-radius: 0 18px 18px 18px;"
        avatar = "🤖"
        flex_direction = "row"
        margin_style = "margin-right: auto;"  # Push to left
        avatar_margin = "margin-right: 8px;"

    # Convert **bold** markdown to HTML
    while "**" in content:
        content = content.replace("**", "<strong>", 1)
        content = content.replace("**", "</strong>", 1)

    # Convert newlines
    html_content = content.replace('\n', '<br>')

    st.markdown(
        f"""
        <div style='display: flex; flex-direction: {flex_direction}; align-items: flex-end; margin-bottom: 10px;'>
            <div style='font-size: 20px; {avatar_margin}'>{avatar}</div>
            <div style='background: {bubble_color}; padding: 10px 14px; width: fit-content;
                        {border} font-family: "Segoe UI", sans-serif;
                        box-shadow: 0 1px 2px rgba(0,0,0,0.1); {margin_style}'>
                {html_content}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Show chat history
for msg in st.session_state.chat_history:
    render_chat_message(msg["role"], msg["content"])

# === Free-form chat input (in addition to menu/buttons) ===
user_input = st.chat_input("Ask me anything about your academic progress, courses, or university policies...")
if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.spinner("🤖 Thinking..."):
        if SELECTED_MODEL == "openai":
            resp = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.chat_history if m["role"] in ("user", "assistant")]
            )
            answer = resp.choices[0].message.content
        else:
            # For local models, concatenate chat history for context (last 4 turns)
            history = st.session_state.chat_history[-4:]
            prompt = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in history]) + "\nAssistant:"
            answer = local_gpt4all(prompt)
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    st.rerun()


# Onboarding message logic
if not st.session_state.greet_done:
    onboarding = (
        "👋 **Welcome to the AI Student Asisstant!**\n\n"
        "I'm here to support your academic journey using smart, personalized guidance. Here's what I can help you with:\n\n"
        "🎓 **Student Info & Progress**\n"
        "- View your **Profile Summary**\n"
        "- Track your **Enrollment History**\n"
        "- See your **Progress by Area of Study**\n"
        "- Check your **Eligible Courses** for next semester\n"
        "- Get **GPT-powered Course Recommendations** tailored to your academic progress\n\n"
        "📘 **Major Sheet Insight**\n"
        "- Understand your **Major Requirements**\n"
        "- Get clear, GPT explanations of **any course** in your major: its role, prerequisites, and importance\n\n"
        "Let’s get started! Choose from the below to begin. 🚀"
    )

    # Save and render immediately
    st.session_state.chat_history.append({"role": "assistant", "content": onboarding})
    st.session_state.greet_done = True
    render_chat_message("assistant", onboarding)
    

# Show choices as buttons
def show_choices(choices):
    n = len(choices)
    cols = st.columns(n)
    for i, label in enumerate(choices):
        with cols[i]:
            if st.button(label, key=f"btn_{label}_{i}", use_container_width=True):
                st.session_state.last_choice = label
                st.rerun()

def show_choices_grid(choices, columns_per_row=2):
    """Render buttons in rows with N buttons per row (centered)."""
    for i in range(0, len(choices), columns_per_row):
        row = choices[i:i+columns_per_row]
        cols = st.columns(columns_per_row, gap="medium")
        for j, label in enumerate(row):
            with cols[j]:
                if st.button(label, key=f"btn_{label}_{i+j}", use_container_width=True):
                    st.session_state.last_choice = label
                    st.rerun()

# === Stage-based UI ===
stage = st.session_state.stage
choice = st.session_state.get("last_choice")

# Menu logic
if st.session_state.stage == "root":
    st.session_state.selected_sid = None  # Reset Student ID when returning to root
    show_choices(["📚 My Academic Info", "📄 Major Sheet"])

elif st.session_state.stage == "my_info":
    if st.session_state.get("selected_sid") is None:
        sid_list = master_df["StudentID"].astype(str).tolist()
        selected_sid = show_selectbox("Please select your Student ID:", sid_list, key="sid_picker")
        if selected_sid:
            st.session_state.selected_sid = selected_sid
            st.session_state.chat_history.append({"role": "user", "content": f"Selected Student ID: {selected_sid}"})
            say_animated("Great! Now choose the information you'd like to see.")
            st.rerun()
    else:
        show_choices_grid([
            "Profile Summary", "Enrollment History",
            "Progress by Area", "Eligible Courses",
            "Eligible Courses (Offered Next Semester)",
            "Course Recommendations", "⬅️ Back"
        ])

elif st.session_state.stage == "major_sheet":
    show_choices(["Course Info", "Major Requirements", "⬅️ Back"])

elif st.session_state.stage == "pick_sid":
    sid_list = master_df["StudentID"].astype(str).tolist()
    selected_sid = show_selectbox("Select Student ID", sid_list, key="sid_picker")
    if selected_sid:
        st.session_state.selected_sid = selected_sid
        st.session_state.stage = "my_info"
        say_animated("Great! Now choose the information you'd like to see.")
        st.rerun()
        
elif st.session_state.stage == "pick_major":
    majors = sorted(set(major_map.values()))
    selected_major = show_selectbox("Select Major", majors, key="major_picker")
    if selected_major:
        st.session_state.last_choice = selected_major
        st.rerun()

elif st.session_state.stage == "pick_course":
    mcode = st.session_state.get("major_code", None)
    if mcode:
        courses = sorted(df_major[df_major["Major"] == mcode]["Course_ID"].dropna().unique())
        selected_course = show_selectbox("Select Course", courses, key="course_picker")
        if selected_course:
            st.session_state.last_choice = selected_course
            st.rerun()
            
# Handle click logic
if choice:
    st.session_state.chat_history.append({"role": "user", "content": choice})

    if stage == "root":
        if choice == "📚 My Academic Info":
            st.session_state.stage = "my_info"
            say_animated("Please select your Student ID to continue.")  # New greeting
        elif choice == "📄 Major Sheet":
            st.session_state.stage = "major_sheet"
            say_animated("Alright! Choose an option:")

    elif stage == "my_info":
        if choice == "⬅️ Back":
            st.session_state.stage = "root"
            st.session_state.selected_sid = None  # Clear selection when going back
            say_animated("Back to main menu.")
        else:
            sid = st.session_state.selected_sid
            action = choice

            if action == "Profile Summary":
                reply = student_profile_text(sid)
            elif action == "Enrollment History":
                reply = enrolled_text(sid)
            elif action == "Progress by Area":
                reply = student_progress_text(sid)
            elif action == "Eligible Courses":
                completed = progress_df[
                    (progress_df["Student_ID"] == sid) &
                    (progress_df["Remaining_Courses"] <= 0)
                ]["AREA_OF_STUDY"].tolist()
                reply = eligible_text(sid, completed)
            elif action == "Eligible Courses (Offered Next Semester)":
                completed = progress_df[
                    (progress_df["Student_ID"] == sid) &
                    (progress_df["Remaining_Courses"] <= 0)
                ]["AREA_OF_STUDY"].tolist()

                df = elig_det[
                    (elig_det["Student_ID"] == sid) & 
                    (~elig_det["AREA_OF_STUDY"].isin(completed))
                ]

                offered_now = schedule_df[schedule_df["Semester"] == current_semester]["CourseID"].unique().tolist()
                df = df[df["Eligible_Courses"].isin(offered_now)]

                if df.empty:
                    reply = "✅ No eligible courses are being offered in the next semester."
                else:
                    reply = "✅ **Eligible Courses Offered in Next Semester:**\n\n"
                    for area, grp in df.groupby("AREA_OF_STUDY"):
                        reply += f"**{area}** ({area_map.get(area, area)}):\n"
                        reply += "- " + " \n- ".join(grp["Eligible_Courses"].tolist()) + "\n\n"
            elif action == "Course Recommendations":
                offered = offered_this_term(sid)
                if not offered:
                    reply = "❌ No eligible courses are offered this semester."
                else:
                    summary = (
                        student_profile_text(sid) + "\n\n" +
                        student_progress_text(sid)
                    )
                    major = master_df.loc[master_df["StudentID"] == sid, "Major"].values[0]
                    with st.spinner("🤖 GPT thinking..."):
                        reply = gpt_recommend_courses(sid, major, summary, offered)

            say_animated(reply)


    elif stage == "major_sheet":
        if choice == "⬅️ Back":
            st.session_state.stage = "root"
            say_animated("Back to main menu.")
        else:
            st.session_state.major_action = choice
            st.session_state.stage = "pick_major"
            say_animated("Please select your major:")

    elif stage == "pick_major":
        st.session_state.selected_major = choice
        mcode = rev_map[choice]
        st.session_state.major_code = mcode

        if st.session_state.major_action == "Course Info":
            st.session_state.stage = "pick_course"
            say_animated("Please select a course:")
        else:
            reqs = req_df[req_df["Major"] == choice]
            reply = f"📋 **Requirements for {choice}:**\n\n"
            for _, r in reqs.iterrows():
                a = r["AREA_OF_STUDY"]
                reply += (
                    f"• **{a}** ({area_map.get(a, a)}) – "
                    f"{int(r['Required_Courses'])} course(s), "
                    f"{int(r['Required_Credits'])} credit(s)\n\n"
                )
            say_animated(reply)
            st.session_state.stage = "major_sheet"

    elif stage == "pick_course":
        course = choice
        row = df_major[
            (df_major["Major"] == st.session_state.major_code) &
            (df_major["Course_ID"] == course)
        ].iloc[0]
        with st.spinner("🤖 GPT thinking..."):
            reply = gpt_explain_course(row.to_dict(), st.session_state.selected_major)

            say_animated(reply)
        st.session_state.stage = "major_sheet"

    st.session_state.last_choice = None
    st.rerun()

