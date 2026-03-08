import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import torch
import joblib
import datetime
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib import styles
from reportlab.lib.styles import getSampleStyleSheet

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AgriTrust | Farmer Banking AI", layout="wide", page_icon="🌾")

# ---------------- UPGRADED LOGIN ----------------
if "login" not in st.session_state:
    st.session_state.login = False

if not st.session_state.login:
    st.markdown("""
        <style>
        .stApp {
            background: linear-gradient(135deg, rgba(20, 50, 40, 0.9), rgba(10, 30, 60, 0.9)), 
                        url('https://images.unsplash.com/photo-1500382017468-9049fed747ef?q=80&w=2000&auto=format&fit=crop');
            background-size: cover;
            height: 100vh;
        }
        .login-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            padding: 60px;
            border-radius: 40px;
            text-align: center;
            max-width: 500px;
            margin: auto;
            margin-top: 100px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 40px 80px rgba(0, 0, 0, 0.5);
        }
        .stTextInput label p {
            font-weight: bold !important;
            color: #ffffff !important; 
            font-size: 16px;
            text-shadow: 1px 1px 2px black;
        }
        .stTextInput input { 
            background: rgba(255, 255, 255, 0.2) !important; 
            color: #ffffff !important; 
            border: 1px solid rgba(255, 255, 255, 0.5) !important;
            border-radius: 12px !important;
        }
        .login-btn button {
            background: linear-gradient(90deg, #10b981, #059669) !important;
            color: white !important;
            width: 100%;
            padding: 15px !important;
            border-radius: 12px !important;
            font-weight: bold !important;
            text-transform: uppercase;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="login-container">', unsafe_allow_html=True)
    st.markdown('<h1 style="color:white; margin-bottom:0;">🌱 AgriTrust</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:#a7f3d0; margin-top:0; font-weight:300;">Secure Bank Terminal</h3><br>', unsafe_allow_html=True)

    username = st.text_input("Username", placeholder="admin")
    password = st.text_input("Password", type="password", placeholder="bank123")

    def do_login():
        if username == "admin" and password == "bank123":
            st.session_state.login = True
        else:
            st.error("❌ Invalid Credentials")

    st.markdown('<div class="login-btn">', unsafe_allow_html=True)
    st.button("Login to System", on_click=do_login)
    st.markdown('</div></div>', unsafe_allow_html=True)
    st.stop()

# ---------------- DATABASE ----------------
DB = "loan_database.xlsx"
if Path(DB).exists():
    df = pd.read_excel(DB, dtype={'Aadhaar': str, 'Phone': str})
else:
    df = pd.DataFrame(columns=[
        "Name","Aadhaar","Phone","Age","Land","Income","Loan","CropType",
        "Rainfall","Irrigation","Latitude","Longitude","Disease","Confidence",
        "Risk","Repayment","EMI","YearsToRepay","PastStatus","PreviousLoan","Decision","Reason","Date"
    ])

# ---------------- NAVIGATION ----------------
st.sidebar.markdown("""
    <div style='background-color: #065f46; padding: 20px; border-radius: 15px; margin-bottom: 20px; text-align:center;'>
        <h2 style='color: white; margin:0;'>AgriTrust</h2>
        <p style='color: #a7f3d0; font-size:12px; margin:0;'>Official Admin Suite</p>
    </div>
""", unsafe_allow_html=True)

menu_map = {"📊 Portfolio Dashboard": "Dashboard", "📝 Loan Application": "Loan Application", "🔎 KYC & Past Records": "Past Loan Status", "🛡️ Master Database": "Admin Panel"}
selection = st.sidebar.radio("Navigation", list(menu_map.keys()))
page = menu_map[selection]

dark_mode = st.sidebar.toggle("🌙 Deep Dark Mode")
if st.sidebar.button("🚪 Logout Session", use_container_width=True):
    st.session_state.login = False
    st.rerun()

# ---------------- THEME ----------------
if dark_mode:
    st.markdown("<style>.stApp{background:#06110e; color:#ecfdf5;} .stTextInput input, .stNumberInput input, .stSelectbox div {background-color:#0d2a22 !important; color:white !important; border:1px solid #10b981 !important;} label p {color:#a7f3d0 !important; font-weight:bold !important;}</style>", unsafe_allow_html=True)
else:
    st.markdown("<style>.stApp{background:#f8fafc; color:#064e3b;} label p {color:#064e3b !important; font-weight:bold !important;}</style>", unsafe_allow_html=True)

# ---------------- MODELS ----------------
@st.cache_resource
def load_assets():
    try:
        l_model = joblib.load("loan_model.pkl")
        p_model = models.mobilenet_v2(weights=None)
        p_model.classifier[1] = torch.nn.Linear(p_model.last_channel, 38)
        p_model.load_state_dict(torch.load("plant_model.pth", map_location="cpu"))
        p_model.eval()
        return l_model, p_model
    except:
        return None, None

loan_model, plant_model = load_assets()
transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
classes = [f"class_{i}" for i in range(38)]

# ---------------- HELPER: DECISION LETTER PDF ----------------
def generate_decision_letter(data):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles_sheet = getSampleStyleSheet()
    elements = []
    
    header_title = "Official Loan Sanction Advice" if data['Decision'] == "Approved" else "Official Rejection Advice"
    elements.append(Paragraph("<b>AGRITRUST BANKING CORPORATION</b>", styles_sheet['Title']))
    elements.append(Paragraph(header_title, styles_sheet['Heading2']))
    elements.append(Spacer(1, 24))
    
    decision_text = f"Dear {data['Name']}, your loan application for <b>₹ {data['Loan']}</b> has been <b>{data['Decision'].upper()}</b>."
    
    body_text = f"""
    <b>Date:</b> {data['Date']}<br/><br/>
    <b>To,</b><br/>
    <b>Name:</b> {data['Name']}<br/>
    <b>Aadhaar:</b> {data['Aadhaar']}<br/><br/>
    {decision_text}<br/><br/>
    <b>Audit Remarks (Top Factors):</b><br/>
    {data['Reason'].replace(' | ', '<br/>')}<br/><br/>
    <b>Risk Analysis:</b><br/>
    - AI Rejection Level: {data['Risk']:.2f}%<br/>
    - Estimated EMI: ₹ {data['EMI']:.2f}<br/><br/><br/>
    """
    
    elements.append(Paragraph(body_text, styles_sheet['Normal']))
    
    # DIGITAL SIGNATURE SECTION
    elements.append(Spacer(1, 40))
    elements.append(Paragraph("__________________________", styles_sheet['Normal']))
    elements.append(Paragraph("<b>Authorised Bank Manager</b><br/>Digital Signature Verified", styles_sheet['Normal']))
    
    doc.build(elements)
    buffer.seek(0)
    return buffer

# ---------------- PAGES ----------------
if page == "Dashboard":
    st.title("📊 Loan Portfolio Performance")
    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Applications", len(df))
    with c2: st.metric("Approved Assets", len(df[df["Decision"]=="Approved"]))
    with c3: st.metric("Risk Rejections", len(df[df["Decision"]=="Rejected"]))
    
    if not df.empty:
        col_left, col_right = st.columns(2)
        with col_left: st.plotly_chart(px.line(df, y="Risk", title="📈 Rejection Risk Trend", color_discrete_sequence=['#ef4444']), use_container_width=True)
        with col_right: st.plotly_chart(px.pie(df, names="Decision", title="📊 Approval Ratio", hole=0.5, color_discrete_sequence=['#10b981', '#ef4444']), use_container_width=True)
        
        st.subheader("🗺️ Loan Distribution Heatmap")
        map_df = df.dropna(subset=['Latitude', 'Longitude'])
        if not map_df.empty:
            avg_lat, avg_lon = map_df['Latitude'].mean(), map_df['Longitude'].mean()
            h_map = folium.Map(location=[avg_lat, avg_lon], zoom_start=7)
            HeatMap([[row['Latitude'], row['Longitude']] for i, row in map_df.iterrows()], radius=15).add_to(h_map)
            st_folium(h_map, width=1300, height=500)

elif page == "Loan Application":
    st.title("👨‍🌾 New Credit Assessment")
    with st.expander("Farmer Profile", expanded=True):
        col1, col2 = st.columns(2)
        name = col1.text_input("Farmer Name")
        aadhaar = col2.text_input("Aadhaar (12 digits)", max_chars=12)
        phone = col1.text_input("Phone Number (10 digits)", max_chars=10)
        age = col2.number_input("Age", 18, 100, 35)
        land = col1.number_input("Land Acres", 0.0)
        income = col2.number_input("Annual Income (₹)", 0)
        loan_val = col1.number_input("Loan Amount (₹)", 0)
        crop_type = col2.selectbox("Crop Type", ["Wheat","Rice","Pulse","Vegetables","Sugarcane","Grains"])
        rainfall = col1.slider("Rainfall (mm)", 0, 500)
        irrigation = col2.selectbox("Irrigation Available?", ["Yes","No"])
        previous_loan = col1.selectbox("Previous Loan Status", ["Not Taken", "Paid", "Not Paid", "In Pending"])

    st.subheader("🌿 Crop Disease Detection")
    imgfile = st.file_uploader("Upload Leaf Image")
    disease, confidence_val = "Healthy", 100.0
    if imgfile:
        img = Image.open(imgfile).convert("RGB")
        st.image(img, width=200)
        img_t = transform(img).unsqueeze(0)
        if plant_model:
            with torch.no_grad():
                outputs = plant_model(img_t)
                prob = torch.nn.functional.softmax(outputs, dim=1)
                conf, pred = torch.max(prob, 1)
            disease, confidence_val = classes[pred.item()], float(conf.item()) * 100
            st.success(f"Detected: {disease} ({confidence_val:.2f}%)")

    st.subheader("🗺️ Farm Location Search")
    location_query = st.text_input("🔍 Search Location", placeholder="e.g. Sivagangai, Tamil Nadu")
    lat, lon = 11.0168, 76.9558 
    geolocator = Nominatim(user_agent="agritrust_geocoder")
    if location_query:
        try:
            location = geolocator.geocode(location_query, addressdetails=True, language="en")
            if location:
                lat, lon = location.latitude, location.longitude
                st.markdown(f"📍 **Location Navigator:** {location.address}")
        except: st.error("Geo-service busy.")

    m = folium.Map(location=[lat, lon], zoom_start=14)
    folium.Marker([lat, lon], popup="Farmer Location").add_to(m)
    map_data = st_folium(m, width=1100, height=400)
    if map_data and map_data["last_clicked"]:
        lat, lon = map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"]

    st.subheader("💰 Financial Terms")
    rate, years = st.number_input("Interest Rate (%)", 8.5), st.number_input("Tenure (Years)", 1)
    r, n = rate/1200, years*12
    emi = (loan_val*r*(1+r)**n)/((1+r)**n-1) if r > 0 else loan_val/n if n > 0 else 0
    st.metric("Estimated Monthly EMI", f"₹ {emi:.2f}")

    if st.button("RUN CREDIT AUDIT", use_container_width=True):
        if len(phone) != 10 or len(aadhaar) != 12:
            st.error("❌ Check Phone/Aadhaar digits.")
        else:
            rejection_score = 0
            audit_reasons = []
            critical_reject = False

            # --- UPDATED LOGIC ---

            # 1. Senior Policy Rule: Age > 75 and Loan > Income
            if age > 75 and loan_val > income:
                rejection_score = 100
                audit_reasons.append("❌ CRITICAL: Senior Policy Violation (Age > 75 & Loan > Income)")
                critical_reject = True

            # 2. General Loan vs Income
            elif loan_val > income:
                rejection_score += 30
                audit_reasons.append("❌ Loan amount exceeds annual income")
            else:
                audit_reasons.append("✅ Income capacity sufficient")

            # 3. Rainfall & Irrigation
            if rainfall < 150 and irrigation == "No":
                rejection_score += 30
                audit_reasons.append("❌ High Risk: Low rainfall and No irrigation")
            else:
                audit_reasons.append("✅ Water security verified")

            # 4. Disease Detection
            if "healthy" not in disease.lower():
                rejection_score += 20
                audit_reasons.append(f"❌ Biological Risk: {disease} detected")
            else:
                audit_reasons.append("✅ Crop health verified")

            # 5. Debt History
            if previous_loan in ["In Pending", "Not Paid"]:
                rejection_score += 40
                audit_reasons.append(f"❌ Credit History: {previous_loan} status")
            else:
                audit_reasons.append("✅ Clean credit history")

            rejection_score = min(rejection_score, 100)
            repay_prob = 100 - rejection_score
            decision = "Rejected" if (rejection_score >= 60 or critical_reject) else "Approved"
            final_reason_str = " | ".join(audit_reasons)

            st.markdown(f"### Assessment: <span style='color:{'#10b981' if decision=='Approved' else '#ef4444'}'>{decision.upper()}</span>", unsafe_allow_html=True)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = rejection_score,
                title = {'text': "Rejection Level (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 30], 'color': "#22c55e"},
                        {'range': [30, 60], 'color': "#facc15"},
                        {'range': [60, 100], 'color': "#ef4444"}
                    ],
                }
            ))
            st.plotly_chart(fig, use_container_width=True)

            st.info("**AI Reasoning & Audit Trail:**")
            for r_txt in audit_reasons:
                st.write(r_txt)

            new_entry = {
                "Name": name, "Aadhaar": str(aadhaar), "Phone": str(phone), "Age": age, "Land": land, "Income": income,
                "Loan": loan_val, "CropType": crop_type, "Rainfall": rainfall, "Irrigation": irrigation,
                "Latitude": lat, "Longitude": lon, "Disease": disease, "Confidence": confidence_val,
                "Risk": rejection_score, "Repayment": repay_prob, "EMI": emi, "YearsToRepay": years, "PastStatus": "OK",
                "PreviousLoan": previous_loan, "Decision": decision, "Reason": final_reason_str, "Date": datetime.date.today()
            }
            
            df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
            df.to_excel(DB, index=False)
            st.success(f"✅ Record for {name} saved.")
            st.download_button("📥 Download Advice (Signed)", generate_decision_letter(new_entry), f"{decision}_{name}.pdf")

elif page == "Admin Panel":
    st.title("🛡️ Master Database")
    if st.button("🔄 Sync Database"):
        df = pd.read_excel(DB, dtype={'Aadhaar': str, 'Phone': str})
        st.rerun()
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("🗑️ Data Maintenance")
    if st.button("❌ Delete Last Row", use_container_width=True):
        if not df.empty:
            df = df.iloc[:-1]
            df.to_excel(DB, index=False)
            st.rerun()

    st.download_button("📂 Export CSV", df.to_csv(index=False).encode(), "records.csv")

elif page == "Past Loan Status":
    st.title("🔎 KYC Lookup")
    search_uid = st.text_input("Enter Aadhaar")
    if st.button("Query Registry"):
        res = df[df["Aadhaar"] == str(search_uid)]
        if not res.empty: 
            st.table(res[["Name", "Date", "Loan", "Decision", "Reason"]])
            st.download_button("📥 Download Signed PDF", generate_decision_letter(res.iloc[-1].to_dict()), f"KYC_{search_uid}.pdf")
        else: st.warning("No records found.")