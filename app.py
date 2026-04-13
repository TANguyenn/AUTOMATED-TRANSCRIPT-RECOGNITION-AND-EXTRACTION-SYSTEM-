# --- Imports ---
import os
import torch
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
from huggingface_hub import snapshot_download
import numpy as np

# --- Khởi tạo session ---
for key in [
    "cropped_tables", "cropped_cells", "selected_area",
    "csv_editor_df", "original_csv_df", "cropped_table", "cell_images"
]:
    if key not in st.session_state:
        if key.endswith("_df"):
            st.session_state[key] = pd.DataFrame()
        elif key.endswith("_table") or key.endswith("_image"):
            st.session_state[key] = None
        else:
            st.session_state[key] = [] if 'cropped' in key else None


# --- Load model Table DETR ---
@st.cache_resource
def load_detr_model(model_name):
    processor = DetrImageProcessor.from_pretrained("microsoft/table-transformer-detection")
    model = TableTransformerForObjectDetection.from_pretrained(model_name)
    return processor, model

# --- Load VietOCR ---
@st.cache_resource
def load_vietocr_model():
    model_path = snapshot_download(repo_id="10Ngoc/task03ocr")
    config = Cfg.load_config_from_file(f"{model_path}/config.json")
    config['weights'] = os.path.join(model_path, "pytorch_model.bin")
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return Predictor(config)

# --- Vẽ box ---
def draw_boxes(image, outputs, processor, model, threshold=0.5):
    draw = ImageDraw.Draw(image)
    width, height = image.size
    target_sizes = torch.tensor([[height, width]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]

    detected_boxes = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        label_name = model.config.id2label[label.item()]
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], box[1]-10), f"{label_name} {score:.2f}", fill="red")
        detected_boxes.append({"label": label_name, "score": score.item(), "box": box})
    return image, detected_boxes

# --- Hàm chính để xử lý bảng ---
def extract_table_to_csv(image, processor_struct, model_struct, detector):
    inputs = processor_struct(images=image, return_tensors="pt").to(model_struct.device)
    with torch.no_grad():
        outputs = model_struct(**inputs)

    scores = outputs.logits.softmax(-1)[0, :, :-1].max(-1).values
    boxes = outputs.pred_boxes[0]
    keep = scores > 0.5
    boxes = boxes[keep]

    areas = [(b[2] * b[3]).item() for b in boxes]
    if areas:
        max_idx = torch.tensor(areas).argmax()
        boxes = torch.cat([boxes[:max_idx], boxes[max_idx+1:]])

    abs_boxes = []
    for box in boxes:
        cx, cy, w, h = box
        x0 = int((cx - w / 2) * image.width)
        y0 = int((cy - h / 2) * image.height)
        x1 = int((cx + w / 2) * image.width)
        y1 = int((cy + h / 2) * image.height)
        abs_boxes.append([x0, y0, x1, y1])

    vis_img = image.copy()
    draw = ImageDraw.Draw(vis_img)
    for box in abs_boxes:
        draw.rectangle(box, outline="green", width=1)

    x_coords = sorted(set([x0 for x0, y0, x1, y1 in abs_boxes] + [x1 for x0, y0, x1, y1 in abs_boxes]))
    y_coords = sorted(set([y0 for x0, y0, x1, y1 in abs_boxes] + [y1 for x0, y0, x1, y1 in abs_boxes]))

    def merge_close(coords, thresh=10):
        merged = []
        for c in coords:
            if not merged or abs(c - merged[-1]) > thresh:
                merged.append(c)
        return merged

    x_lines = merge_close(x_coords)
    y_lines = merge_close(y_coords)

    if len(x_lines) < 2 or len(y_lines) < 2:
        return None, None, None, None, "⚠️ Không đủ đường kẻ để tạo bảng."

    PAD = 2
    data, cell_images = [], []
    for row in range(len(y_lines) - 1):
        row_data = []
        for col in range(len(x_lines) - 1):
            x0 = max(x_lines[col] - PAD, 0)
            y0 = max(y_lines[row] - PAD, 0)
            x1 = min(x_lines[col + 1] + PAD, image.width)
            y1 = min(y_lines[row + 1] + PAD, image.height)

            if x1 <= x0 or y1 <= y0:
                row_data.append("")
                cell_images.append((None, ""))
                continue

            cell_img = image.crop((x0, y0, x1, y1))
            try:
                txt = detector.predict(cell_img).strip().replace("#", " ")
            except:
                txt = ""
            row_data.append(txt)
            cell_images.append((cell_img, txt))

            draw.rectangle([x0, y0, x1, y1], outline="blue", width=1)
        data.append(row_data)

    df = pd.DataFrame(data)

    # --- Xử lý ký tự '.' ---
    def clean_multiple_dots(value):
        if isinstance(value, str):
            dot_indices = [i for i, c in enumerate(value) if c == "."]
            if len(dot_indices) > 1:
                chars = list(value)
                for idx in dot_indices[1:]:
                    chars[idx] = ''
                return ''.join(chars)
        return value

    df = df.applymap(clean_multiple_dots)

    csv_data = df.to_csv(index=False)
    return df, csv_data, vis_img, cell_images, None

# --- Giao diện Streamlit ---
st.set_page_config(page_title='Automation Tools', layout='wide', page_icon='./images/icon_1.png')
st.image('./images/BG.png', use_column_width=True)
st.markdown("""<style>.centered { text-align: center; }</style>""", unsafe_allow_html=True)
st.markdown('<h1 class="centered">TỰ ĐỘNG NHẬP LIỆU BẰNG OCR</h1>', unsafe_allow_html=True)

with st.expander("Thông tin đề tài"):
    st.markdown('<div class="centered"><strong>KHÓA LUẬN TỐT NGHIỆP 2025</strong></div>', unsafe_allow_html=True)
    st.markdown('<div class="centered"><strong>GVHD: TS. TRẦN NHẬT QUANG</strong></div>', unsafe_allow_html=True)
    st.markdown('<div class="centered"><strong>SV: VĂN THỊ MƯỜI NGỌC - NGUYỄN THỊ NHẬT VY</strong></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Tải ảnh cần kiểm tra", type=["jpg", "png", "jpeg"])
model_choice = st.sidebar.selectbox("📌 Chọn mô hình", [
    "10Ngoc/task01tableDec", 
    "10Ngoc/task02update",
    "10Ngoc/task03ocr",
    "task04csvexport"
])

if "csv_editor_df" not in st.session_state:
    st.session_state.csv_editor_df = pd.DataFrame()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Ảnh gốc", use_column_width=True)

    if model_choice == "task04csvexport":
        # --- Chạy model nếu bấm nút hoặc chưa có bảng ---
        if st.button("🔍 Export to CSV") or st.session_state.csv_editor_df.empty:
            st.info("📋 Đang xử lý bảng...")

            processor_table, model_table = load_detr_model("10Ngoc/task01tableDec")
            processor_struct, model_struct = load_detr_model("10Ngoc/task02update")
            detector = load_vietocr_model()

            inputs = processor_table(images=image, return_tensors="pt").to(model_table.device)
            with torch.no_grad():
                outputs = model_table(**inputs)
            scores = outputs.logits.softmax(-1)[0, :, :-1].max(-1).values
            boxes = outputs.pred_boxes[0]
            keep = scores > 0.5
            boxes = boxes[keep]

            if len(boxes) == 0:
                st.warning("⚠️ Không tìm thấy bảng trong ảnh.")
            else:
                areas = [(b[2]*b[3]).item() for b in boxes]
                largest_idx = torch.tensor(areas).argmax()
                table_box = boxes[largest_idx]
                cx, cy, w, h = table_box
                x0 = int((cx - w / 2) * image.width)
                y0 = int((cy - h / 2) * image.height)
                x1 = int((cx + w / 2) * image.width)
                y1 = int((cy + h / 2) * image.height)
                cropped_image = image.crop((x0, y0, x1, y1))

                df, csv_data, cropped_table, cell_images, err_msg = extract_table_to_csv(
                    cropped_image, processor_struct, model_struct, detector
                )

                if df is not None and df.shape[0] > 1:
                    for col in df.columns:
                        if str(df.iloc[0][col]).strip().lower() == "chữ ký":
                            df.loc[1:, col] = ""

                if err_msg:
                    st.warning(err_msg)
                else:
                    st.session_state.csv_editor_df = df
                    st.session_state.original_csv_df = df.copy()
                    st.session_state.cropped_table = cropped_image
                    st.session_state.cell_images = cell_images

        # --- Hiển thị bảng và cho phép chỉnh sửa ---
        if "csv_editor_df" in st.session_state:
            if "cropped_table" in st.session_state and st.session_state.cropped_table is not None:
                st.image(st.session_state.cropped_table, caption="🧾 Bảng đã cắt")



            with st.form("csv_editor_form", clear_on_submit=False):
                st.markdown("### 📝 Chỉnh sửa bảng CSV")
                edited_df = st.data_editor(
                    st.session_state.csv_editor_df,
                    num_rows="dynamic",
                    use_container_width=True,
                    key="editor"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    if st.form_submit_button("➕ Thêm dòng"):
                        new_row = pd.Series([""] * edited_df.shape[1], index=edited_df.columns)
                        st.session_state.csv_editor_df = pd.concat(
                            [edited_df, pd.DataFrame([new_row])], ignore_index=True
                        )

                with col2:
                    row_to_delete = st.number_input("🗑️ Xoá dòng", min_value=0,
                                                    max_value=max(0, len(edited_df) - 1), step=1)
                    if st.form_submit_button("Xoá dòng"):
                        st.session_state.csv_editor_df = edited_df.drop(index=row_to_delete).reset_index(drop=True)

                with col3:
                    col_to_delete = st.selectbox("🗑️ Xoá cột", options=list(edited_df.columns))
                    if st.form_submit_button("Xoá cột"):
                        st.session_state.csv_editor_df = edited_df.drop(columns=[col_to_delete])

                with col4:
                    new_col_name = st.text_input("➕ Tên cột mới", value="NewCol")
                    if st.form_submit_button("Thêm cột"):
                        edited_df[new_col_name] = ""
                        st.session_state.csv_editor_df = edited_df

            # Tải CSV sau chỉnh sửa
            st.download_button(
                "📥 Tải CSV đã chỉnh",
                data=st.session_state.csv_editor_df.to_csv(index=False),
                file_name="edited_result.csv",
                mime="text/csv"
            )

            # ✅ Nút Reset bảng
            if st.button("🔄 Reset bảng CSV"):
                if not st.session_state.original_csv_df.empty:
                    st.session_state.csv_editor_df = st.session_state.original_csv_df.copy()
                    st.success("✅ Đã khôi phục bảng về trạng thái ban đầu.")
                else:
                    st.warning("⚠️ Chưa có bảng gốc để khôi phục.")


    elif model_choice in ["10Ngoc/task01tableDec", "10Ngoc/task02update"]:
        if st.button("🔍 Detect"):
            processor, model = load_detr_model(model_choice)
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            result_img = image.copy()
            result_img, boxes = draw_boxes(result_img, outputs, processor, model)
            st.image(result_img, caption="🧠 Kết quả detect")

    elif model_choice == "10Ngoc/task03ocr":
        if st.button("🔍 OCR vùng"):
            detector = load_vietocr_model()
            crop = image.crop(st.session_state.selected_area)
            st.image(crop, caption="🗈️ Vùng đã chọn")
            text = detector.predict(crop)
            st.success(f"📖 Kết quả OCR: {text}")

    if model_choice != "task04csvexport":
        if st.button("🔄 Reset"):
            for k in ["cropped_tables", "cropped_cells", "result_image", "selected_area"]:
                st.session_state[k] = [] if 'cropped' in k else None
            st.success("✅ Đã reset!")

with st.expander('📬 Liên hệ'):
    with st.form(key='contact', clear_on_submit=True):
        email = st.text_input('Email')
        text = st.text_area('Nội dung')
        if st.form_submit_button('Gửi') and email:
            os.makedirs('contacts', exist_ok=True)
            with open(f"contacts/{email}.txt", "w", encoding="utf-8") as f:
                f.write(text)
            st.success("📩 Đã gửi! Vui lòng kiểm tra email.")