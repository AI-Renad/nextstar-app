import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import re
from streamlit_echarts import st_echarts  # مكتبة Radar Chart

# يجب وضع set_page_config هنا أولاً
st.set_page_config(page_title="NextStar", layout="wide", initial_sidebar_state="expanded")
# 💜 رأس الصفحة

def render_header():
    st.markdown("""
        <div style='text-align: center;'>
            <img src='https://media.giphy.com/media/SYELZgPZz2V4RzFg6Z/giphy.gif' width='100'/>
            <h1 style='color:#FF00FF;'>NextStar ⚽</h1>
            <p style='color:#fff; font-size: 18px;'>منصة ذكية لتقييم المواهب الكروية بالفيديو والبيانات</p>
            <hr style='border: 2px solid #fff;'>
        </div>
    """, unsafe_allow_html=True)

# ===============================
# 💫 NextStar: Player Analyzer AI
# توثيق شامل
"""
تطبيق NextStar لتحليل أداء اللاعبين باستخدام الذكاء الاصطناعي وملفات الفيديو.
"""

# ===============================
# 💜 عنوان موحّد لكل الصفحات مع تأثيرات الزينة
def render_header():
    st.markdown("""
        <div style='background-color: #8A2BE2; padding: 20px; border-radius: 15px; box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);'>
            <h1 style='text-align: center; color: #FF00FF; text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);'>NextStar ⚽</h1>
            <p style='text-align: center; color: #FF00FF; font-size: 18px; text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.3);'>
                منصة ذكية لتقييم المواهب الكروية بالفيديو والبيانات
            </p>
        </div>
        <hr style="border: 1px solid #8A2BE2; margin-top: 10px;">
    """, unsafe_allow_html=True)

# ===============================
# ⚙️ تحويل القيم النصية إلى أرقام
def convert_value_to_numeric(value):
    value = str(value).replace('\x80', '').replace('€', '').replace(',', '')
    if 'M' in value:
        value = value.replace('M', '').strip()
        try:
            return float(value) * 1_000_000
        except ValueError:
            return np.nan
    elif 'K' in value:
        value = value.replace('K', '').strip()
        try:
            return float(value) * 1_000
        except ValueError:
            return np.nan
    try:
        return float(value)
    except ValueError:
        return np.nan

# ===============================
# 📂 تحميل البيانات
@st.cache_data
def load_player_data():
    df = pd.read_csv("kl.csv", encoding="latin1")
    df['Value'] = df['Value'].apply(convert_value_to_numeric)
    return df

# ===============================
# 🧠 تدريب النموذج
def train_market_model(df):
    model = LinearRegression()
    features = ['Acceleration', 'SprintSpeed', 'Finishing', 'ShotPower', 'LongShots', 'ShortPassing',
                'LongPassing', 'Vision', 'Dribbling', 'BallControl', 'StandingTackle', 'SlidingTackle',
                'Marking', 'Strength']
    df_clean = df.dropna(subset=features + ['Value'])
    model.fit(df_clean[features], df_clean['Value'])
    return model

# ===============================
# 🔮 التنبؤ بالقيمة السوقية
def predict_market_value(model, features_values):
    return model.predict(np.array([features_values]))[0]

# ===============================
# 👤 صفحة تسجيل الدخول مع تصميم الزينة
def login_page():
    render_header()
    st.image("logo.png", width=120)

    with st.form("login_form"):
        name = st.text_input("👤 الاسم", placeholder="أدخل اسمك هنا", help="الرجاء إدخال اسمك")
        email = st.text_input("📧 البريد الإلكتروني", placeholder="أدخل بريدك الإلكتروني", help="الرجاء إدخال بريدك الإلكتروني")
        submitted = st.form_submit_button("دخول", use_container_width=True)

        if submitted:
            if name and email:
                st.session_state.logged_in = True
                st.session_state.user_name = name
                st.success(f"مرحباً {name} 👋", icon="✅")
            else:
                st.error("الرجاء إدخال الاسم والبريد الإلكتروني.", icon="⚠️")

# ===============================
# 🎥 صفحة تحليل الفيديو مع الزينة
def video_analysis_page():
    render_header()

    st.title("تحليل أداء اللاعب من الفيديو", anchor="video_analysis_title")

    video_file = st.file_uploader("🎥 ارفع فيديو اللاعب", type=["mp4", "mov"])
    player_name = st.text_input("اسم اللاعب الموجود في الفيديو")
    player_age = st.number_input("عمر اللاعب", min_value=10, max_value=50)

    if video_file and player_name and player_age:
        st.video(video_file)

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps > 0 else 0

        # تحليل مبسط
        passes = int(duration // 5)
        goals = int(duration // 30)
        tackles = int(duration // 15)

        passes = min(passes, 100)
        goals = min(goals, 10)
        tackles = min(tackles, 30)

        # ميزات إضافية
        additional_features = [
            passes, goals, tackles,  # من الفيديو
            50, 60, 65, 70, 75, 80, 85,
            90, 50, 65, 60
        ]

        if len(additional_features) != 14:
            st.error(f"يجب تقديم 14 ميزة، ولكن تم تقديم {len(additional_features)} ميزة.")
            return

        df = load_player_data()
        model = train_market_model(df)

        try:
            predicted_value = predict_market_value(model, additional_features)
            st.markdown("### 📊 <span style='color:#8A2BE2'>إحصائيات اللاعب في الفيديو:</span>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns(3)
            col1.metric("التمريرات", passes)
            col2.metric("الأهداف", goals)
            col3.metric("التكتلات", tackles)
            st.success(f"💰 القيمة السوقية المتوقعة: {predicted_value:.2f} مليون يورو")

        except ValueError as e:
            st.error(f"خطأ في التنبؤ: {e}")

        st.divider()

        selected_player = st.selectbox("اختر لاعب للمقارنة", df['Name'])
        player_data = df[df['Name'] == selected_player].iloc[0]

        st.markdown(f"### 📌 <span style='color:#FF00FF'>مقارنة بين {player_name} و {selected_player}</span>", unsafe_allow_html=True)
        comparison = pd.DataFrame({
            'المؤشر': ['التمريرات', 'الأهداف', 'التكتلات', 'القيمة السوقية'],
            player_name: [passes, goals, tackles, predicted_value],
            selected_player: [
                player_data['Acceleration'],
                player_data['SprintSpeed'],
                player_data['Finishing'],
                player_data['Value']
            ]
        })

        st.dataframe(comparison.set_index('المؤشر'), use_container_width=True)
        st.bar_chart(comparison.set_index('المؤشر'))

        # إضافة الرادار التفاعلي
        radar_data = {
            "tooltip": {
                "trigger": "item"
            },
            "radar": {
                "indicator": [
                    {"name": "التمريرات", "max": 100},
                    {"name": "الأهداف", "max": 10},
                    {"name": "التكتلات", "max": 30},
                    {"name": "القيمة السوقية", "max": 100},
                    {"name": "السرعة", "max": 100}
                ]
            },
            "series": [{
                "name": "إحصائيات اللاعب",
                "type": "radar",
                "data": [{
                    "value": [passes, goals, tackles, predicted_value, player_data['SprintSpeed']],
                    "name": player_name
                }]
            }]
        }

        st_echarts(options=radar_data)

# ===============================
# 🔍 صفحة استكشاف اللاعبين مع الزينة
def player_exploration_page():
    render_header()

    st.title("استكشاف اللاعبين", anchor="player_exploration_title")
    st.markdown("**فلترة اللاعبين بناءً على العمر، المركز، المهارات، وأكثر...**", unsafe_allow_html=True)

    # تحميل البيانات
    df = load_player_data()

    # اختيار الفلاتر
    age_filter = st.slider("فلتر العمر", 10, 50, (20, 30))
    position_filter = st.selectbox("اختيار المركز", ['كل المراكز'] + df['Position'].unique().tolist())
    skill_filter = st.multiselect(
        "اختيار المهارات",
        ['SprintSpeed', 'Finishing', 'ShotPower', 'LongShots', 'ShortPassing', 'LongPassing', 'Dribbling', 'BallControl'],
        default=['SprintSpeed', 'Finishing']
    )

    # تطبيق الفلاتر
    filtered_df = df[df['Age'].between(age_filter[0], age_filter[1])]

    if position_filter != 'كل المراكز':
        filtered_df = filtered_df[filtered_df['Position'] == position_filter]

    # فلترة المهارات بناءً على المهارات المختارة
    if skill_filter:
        for skill in skill_filter:
            filtered_df = filtered_df[filtered_df[skill] > 0]

    # عرض اللاعبين المصفين
    st.subheader("اللاعبين المصفين:", anchor="filtered_players_subheader")
    st.dataframe(filtered_df[['Name', 'Age', 'Position'] + skill_filter], use_container_width=True)

    st.divider()

# ===============================
# 📊 صفحة Dashboard رئيسي مع الزينة
def dashboard_page():
    render_header()

    st.title("لوحة التحكم الرئيسية", anchor="dashboard_page_title")

    # تحميل البيانات
    df = load_player_data()

    # 1. عدد اللاعبين
    num_players = len(df)
    st.subheader(f"عدد اللاعبين: {num_players}")

    # 2. أكثر مركز
    most_common_position = df['Position'].mode()[0]
    st.subheader(f"أكثر مركز: {most_common_position}")

    # 3. متوسط القيمة السوقية
    avg_market_value = df['Value'].mean()
    st.subheader(f"متوسط القيمة السوقية: {avg_market_value:.2f} يورو")

    # 4. الرسوم البيانية
    st.subheader("الرسوم البيانية")

    # توزيع الأعمار
    st.markdown("### 📊 توزيع الأعمار", unsafe_allow_html=True)
    fig_age, ax_age = plt.subplots()
    sns.histplot(df['Age'], kde=True, ax=ax_age)
    ax_age.set_title("توزيع الأعمار")
    st.pyplot(fig_age)

    # توزيع القيمة السوقية
    st.markdown("### 📊 توزيع القيمة السوقية", unsafe_allow_html=True)
    fig_value, ax_value = plt.subplots()
    sns.histplot(df['Value'], kde=True, ax=ax_value)
    ax_value.set_title("توزيع القيمة السوقية")
    st.pyplot(fig_value)

    # رسم بياني للمركز والقيمة السوقية
    st.markdown("### 📊 المركز والقيمة السوقية", unsafe_allow_html=True)
    fig_position_value, ax_position_value = plt.subplots()
    sns.boxplot(x=df['Position'], y=df['Value'], ax=ax_position_value)
    ax_position_value.set_title("القيمة السوقية حسب المركز")
    ax_position_value.set_xticklabels(ax_position_value.get_xticklabels(), rotation=45)
    st.pyplot(fig_position_value)

    st.divider()

# ===============================
# 🚀 تحديث الصفحة الرئيسية مع الزينة
def main():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login_page()
    else:
        st.sidebar.markdown("<h2 style='color:#8A2BE2;'>📁 التنقل</h2>", unsafe_allow_html=True)
        page = st.sidebar.radio("اذهب إلى:", ["تحليل الفيديو", "استكشاف اللاعبين", "لوحة التحكم", "تسجيل الخروج"])

        if page == "تحليل الفيديو":
            video_analysis_page()
        elif page == "استكشاف اللاعبين":
            player_exploration_page()
        elif page == "لوحة التحكم":
            dashboard_page()
        elif page == "تسجيل الخروج":
            st.session_state.logged_in = False
            st.rerun()

# ===============================
if __name__ == "__main__":
    main()
