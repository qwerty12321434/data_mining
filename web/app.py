# app.py
from flask import Flask, request, render_template
from joblib import load
import pandas as pd
import numpy as np # Sử dụng cho việc làm tròn và giới hạn

# --- Cấu hình Matplotlib (Phải ở trước import plt) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# ---
import io
import base64 # Dùng để mã hóa ảnh, gửi sang HTML
from sklearn.preprocessing import StandardScaler 
from sklearn.cluster import KMeans
import warnings

# Khởi tạo ứng dụng Flask
app = Flask(__name__)

# --- Tải Mô hình Đã Huấn Luyện ---
MODEL_FILE = 'linear_model.joblib'
try:
    # Tải mô hình (chỉ 1 lần khi server khởi động)
    model = load(MODEL_FILE)
    print("Mô hình Linear Regression đã được tải thành công!")
except FileNotFoundError:
    print(f"LỖI: KHÔNG tìm thấy tệp {MODEL_FILE}. Vui lòng kiểm tra lại.")
    model = None

# Tên các Feature (Đảm bảo đúng thứ tự đã train)
FEATURE_COLS = ['G1', 'G2', 'studytime', 'absences', 'failures']

# Hàm đánh giá và phân loại (Dựa trên Business Rules)
def analyze_prediction(predicted_g3, failures, absences, studytime):
    # 1. Phân loại theo Điểm Dự đoán (Thang 20)
    if predicted_g3 > 14:
        score_group = "✅ Thành tích Tốt (Dự kiến G3 > 14)"
    elif predicted_g3 >= 10:
        score_group = "🟡 Trung bình/Ổn định (Dự kiến G3 từ 10 - 14)"
    else:
        score_group = "🚨 Rủi ro Cao (Dự kiến G3 < 10)"

    # 2. Phân tích Yếu tố Hành vi (Risk Factors)
    risk_factors = []
    
    # Rủi ro 1: Lịch sử thất bại
    if failures >= 1:
        risk_factors.append(f"⚠️ Rủi ro Lịch sử: Từng rớt {int(failures)} môn trước.")
    
    # Rủi ro 2: Thiếu kỷ luật (mức vắng cao hơn trung bình ~5.7)
    if absences > 5:
        risk_factors.append(f"⚠️ Rủi ro Kỷ luật: Số buổi vắng cao ({int(absences)} buổi).")
    
    # Rủi ro 3: Hiệu suất học (studytime thấp hoặc cao quá mức)
    if studytime <= 1:
        risk_factors.append("⚠️ Rủi ro Nỗ lực: Thời gian học quá thấp (≤ 2h/tuần).")
    elif studytime >= 4 and predicted_g3 < 12:
        # Phát hiện studytime cao nhưng điểm thấp (vấn đề hiệu suất)
        risk_factors.append("🟡 Phân tích Hiệu suất: Nỗ lực cao (≥ 10h/tuần) nhưng điểm chưa tương xứng (cần cải thiện phương pháp).")
        
    if not risk_factors:
        risk_factors.append("👍 Sinh viên ổn định, không có yếu tố rủi ro hành vi đáng kể.")

    return score_group, risk_factors

# --- Định tuyến (Routing) ---

# Trang chủ - Hiển thị form nhập liệu
@app.route('/')
def home():
    # Render trang HTML, cung cấp giá trị mặc định cho form
    default_values = {'g1': 12, 'g2': 13, 'studytime': 2, 'absences': 4, 'failures': 0}
    return render_template('index.html', **default_values)

# API dự đoán - Xử lý POST request từ form
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Lỗi: Mô hình chưa được tải.", 500
        
    try:
        # Lấy dữ liệu từ form (tất cả đều là string, cần chuyển sang float)
        data = [
            float(request.form['g1']),
            float(request.form['g2']),
            float(request.form['studytime']),
            float(request.form['absences']),
            float(request.form['failures'])
        ]

        # Tạo DataFrame để đảm bảo thứ tự và cấu trúc inputs đúng với mô hình đã train
        input_df = pd.DataFrame([data], columns=FEATURE_COLS)
        
        # Thực hiện dự đoán
        prediction = model.predict(input_df)[0]
        
        # Làm tròn điểm dự đoán và giới hạn trong khoảng [0, 20]
        final_g3 = max(0, min(20, round(prediction)))
        
        # Phân tích kết quả
        score_group, risk_factors = analyze_prediction(
            final_g3, 
            data[4], # failures
            data[3], # absences
            data[2]  # studytime
        )
        
        # Trả kết quả về trang HTML, giữ lại giá trị đã nhập
        return render_template('index.html', 
                               prediction_text=f'{final_g3} / 20',
                               score_group=score_group,
                               risk_factors=risk_factors,
                               g1=data[0], g2=data[1], studytime=data[2], absences=data[3], failures=data[4])

    except ValueError:
        # Xử lý lỗi nếu người dùng nhập ký tự không phải số
        return render_template('index.html', error_message='Dữ liệu nhập vào không hợp lệ. Vui lòng kiểm tra các trường.')
    except Exception as e:
        # Xử lý lỗi hệ thống
        return render_template('index.html', error_message=f'Lỗi hệ thống không xác định: {str(e)}')

# ======================================================================
# --- PHẦN K-Means (Gộp lại) ---
# ======================================================================

# 1. Hàm Tải và Tiền xử lý Dữ liệu
def load_and_preprocess_data():
    file_path = 'data/student-mat.csv' # Đảm bảo thư mục 'data' ngang hàng với 'app.py'
    try:
        df_raw = pd.read_csv(file_path, sep=';')
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file tại '{file_path}'")
        return None
    df = df_raw.copy()
    try:
        df['G1'] = pd.to_numeric(df['G1'])
        df['G2'] = pd.to_numeric(df['G2'])
    except ValueError as e:
        print(f"Lỗi khi chuyển đổi cột điểm: {e}")
        return None
    cols_to_map = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
    for col in cols_to_map:
        if col in df.columns:
            df[col] = df[col].map({'yes': 1, 'no': 0})
    return df

# 2. Hàm Phụ trợ để tạo biểu đồ và chuyển sang Base64
def create_elbow_plot_base64(X_scaled):
    inertia = []
    K_range = range(2, 11)
    for k in K_range:
        kmeans_elbow = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
        inertia.append(kmeans_elbow.inertia_)
    fig, ax = plt.subplots()
    ax.plot(K_range, inertia, 'bo-')
    ax.set_xlabel('Số cụm (K)')
    ax.set_ylabel('Inertia')
    ax.grid(True)
    img = io.BytesIO()
    fig.savefig(img, format='png', bbox_inches='tight')
    img.seek(0)
    plt.close(fig)
    plot_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{plot_base64}"

# 3. *** ĐÃ XÓA ROUTE /clusters ***

# --- PHẦN PHÂN CỤM TÙY CHỈNH (Giờ là trang duy nhất) ---

# 1. Định nghĩa các cột có thể phân tích
NUMERIC_COLS = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 
                'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 
                'absences', 'G1', 'G2', 'G3']
YES_NO_COLS = ['schoolsup', 'famsup', 'paid', 'activities', 'nursery', 
               'higher', 'internet', 'romantic']
ALL_ANALYZABLE_COLS = sorted(NUMERIC_COLS + YES_NO_COLS)

FEATURE_DESCRIPTIONS = {
    'age': 'Tuổi', 'Medu': 'Học vấn mẹ', 'Fedu': 'Học vấn cha',
    'traveltime': 'Thời gian đi lại', 'studytime': 'Thời gian học/tuần',
    'failures': 'Số lần trượt môn', 'famrel': 'Quan hệ gia đình',
    'freetime': 'Thời gian rảnh', 'goout': 'Đi chơi với bạn',
    'Dalc': 'Uống rượu ngày thường', 'Walc': 'Uống rượu cuối tuần',
    'health': 'Sức khỏe', 'absences': 'Số buổi vắng',
    'G1': 'Điểm kỳ 1', 'G2': 'Điểm kỳ 2', 'G3': 'Điểm cuối kỳ',
    'schoolsup': 'Hỗ trợ thêm từ trường', 'famsup': 'Hỗ trợ từ gia đình',
    'paid': 'Học thêm trả phí', 'activities': 'Hoạt động ngoại khóa',
    'nursery': 'Đã đi nhà trẻ', 'higher': 'Muốn học cao hơn',
    'internet': 'Có Internet ở nhà', 'romantic': 'Đang yêu'
}

# 2. Tạo route mới cho trang Tùy chỉnh (Giữ nguyên logic từ lần trước)
@app.route('/interactive', methods=['GET', 'POST'])
def interactive_cluster():
    # Khởi tạo các biến
    results = {}
    error_msg = None
    
    # Khi người dùng GỬI YÊU CẦU (nhấn 1 trong 2 nút)
    if request.method == 'POST':
        try:
            # 1. Lấy data từ form
            action = request.form.get('action') # Lấy tên của nút đã nhấn
            selected_features = request.form.getlist('features') # Lấy list các checkbox
            k = int(request.form.get('k_clusters', 3)) # Lấy K, mặc định là 3

            # "Nhớ" lại các lựa chọn của người dùng để hiển thị lại
            results['selected_cols'] = selected_features
            results['selected_k'] = k
            
            if not selected_features:
                raise ValueError("Bạn chưa chọn bất kỳ feature nào.")

            # --- Chạy các bước chung cho CẢ HAI NÚT ---
            
            # 2. Tải và xử lý data
            df = load_and_preprocess_data()
            if df is None:
                raise Exception("Không thể tải dữ liệu.")
            
            # 3. Lấy đúng các cột đã chọn
            X_custom = df[selected_features].copy()
            
            # 4. Hiển thị bảng data gốc (5 dòng đầu)
            results['selected_data_html'] = X_custom.head().to_html(classes='table table-striped table-hover', justify='center')
            
            # 5. Chuẩn hóa
            scaler = StandardScaler()
            X_custom_scaled = scaler.fit_transform(X_custom)
            
            # 6. Hiển thị bảng data đã chuẩn hóa (5 dòng đầu)
            scaled_df_head = pd.DataFrame(X_custom_scaled, columns=selected_features).head()
            results['scaled_data_html'] = scaled_df_head.round(3).to_html(classes='table table-striped table-hover', justify='center')
            
            # 7. Chạy Elbow (để vẽ)
            results['plot_img'] = create_elbow_plot_base64(X_custom_scaled)
            
            # --- Chỉ chạy bước cuối nếu nhấn nút "Run Cluster" ---
            if action == 'run_cluster':
                if k < 2 or k > 10:
                    raise ValueError("Số cụm (K) phải nằm trong khoảng 2 đến 10.")
                
                # 8. Chạy K-Means (với K người dùng chọn)
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_custom_scaled)
                df['custom_cluster'] = kmeans.labels_
                profile = df.groupby('custom_cluster')[selected_features].mean()
                results['table_html'] = profile.T.round(2).to_html(classes='table table-striped table-hover')
            
            # 'get_k' không cần làm gì thêm, chỉ hiển thị 3 bước trên

        except Exception as e:
            # Nếu lỗi, gửi thông báo lỗi
            error_msg = f"Lỗi Xảy Ra: {e}"
    
    # Khi MỚI VÀO TRANG (GET) hoặc sau khi xử lý (POST)
    return render_template('interactive.html', 
                           all_cols=ALL_ANALYZABLE_COLS, 
                           descriptions=FEATURE_DESCRIPTIONS, 
                           results=results,
                           error_msg=error_msg)

# --- Chạy ứng dụng ---
if __name__ == '__main__':
    # Chạy ứng dụng web
    app.run(debug=True)