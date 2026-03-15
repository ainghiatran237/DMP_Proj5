from __future__ import annotations

import pandas as pd
import streamlit as st

from core.config import SAMPLE_SOURCE_LABEL

DEFAULT_LANGUAGE = "en"
LANGUAGE_OPTIONS = {
    "en": "English",
    "vi": "Tiếng Việt",
}

TRANSLATIONS: dict[str, dict[str, str]] = {
    "app.title": {
        "en": "Airbnb Analytics Dashboard",
        "vi": "Bảng Điều Khiển Phân Tích Airbnb",
    },
    "common.switch_language_help": {
        "en": "Current: {current}. Click to switch to {next}.",
        "vi": "Hiện tại: {current}. Bấm để chuyển sang {next}.",
    },
    "common.language_toggle": {
        "en": "English / Tiếng Việt",
        "vi": "English / Tiếng Việt",
    },
    "common.na": {
        "en": "N/A",
        "vi": "Không có",
    },
    "common.days_suffix": {
        "en": "{value} days",
        "vi": "{value} ngày",
    },
    "common.analysis_mode": {
        "en": "Analysis mode",
        "vi": "Chế độ phân tích",
    },
    "common.live": {
        "en": "Live",
        "vi": "Trực tiếp",
    },
    "source.sample": {
        "en": "Bundled sample data",
        "vi": "Dữ liệu mẫu đi kèm",
    },
    "nav.overview.label": {
        "en": "Overview",
        "vi": "Tổng quan",
    },
    "nav.data_raw.label": {
        "en": "Data Raw",
        "vi": "Dữ liệu gốc",
    },
    "nav.preprocessing.label": {
        "en": "Preprocessing",
        "vi": "Tiền xử lý",
    },
    "nav.eda.label": {
        "en": "EDA Insights",
        "vi": "Phân tích EDA",
    },
    "nav.conclusion.label": {
        "en": "Conclusion",
        "vi": "Kết luận",
    },
    "nav.chatbot.label": {
        "en": "Chatbot",
        "vi": "Trợ lý AI",
    },
    "sidebar.live_badge": {
        "en": "Active",
        "vi": "Đang mở",
    },
    "sidebar.dataset_status": {
        "en": "Dataset status",
        "vi": "Trạng thái dữ liệu",
    },
    "sidebar.dataset_ready": {
        "en": "Prepared counts and source status for this workspace.",
        "vi": "Các chỉ số đã chuẩn bị và trạng thái nguồn cho workspace này.",
    },
    "sidebar.prepared_rows": {
        "en": "Prepared rows",
        "vi": "Dòng đã chuẩn bị",
    },
    "sidebar.neighborhood_groups": {
        "en": "Neighborhood groups",
        "vi": "Nhóm khu vực",
    },
    "sidebar.room_types": {
        "en": "Room types",
        "vi": "Loại phòng",
    },
    "sidebar.navigate": {
        "en": "Navigate",
        "vi": "Điều hướng",
    },
    "sidebar.current_view": {
        "en": "Current view",
        "vi": "Màn hình hiện tại",
    },
    "sidebar.session_actions": {
        "en": "Session actions",
        "vi": "Tác vụ phiên",
    },
    "sidebar.logout_hint": {
        "en": "Leave the current workspace and return to the sign-in flow.",
        "vi": "Rời workspace hiện tại và quay về luồng đăng nhập.",
    },
    "sidebar.logout": {
        "en": "Logout session",
        "vi": "Đăng xuất",
    },
    "auth.language_hint": {
        "en": "Display language",
        "vi": "Ngôn ngữ hiển thị",
    },
    "login.badge": {
        "en": "Airbnb analytics workspace",
        "vi": "Không gian phân tích Airbnb",
    },
    "login.title": {
        "en": "Sign in and continue reading the market.",
        "vi": "Đăng nhập để tiếp tục đọc bức tranh thị trường.",
    },
    "login.body": {
        "en": "Explore pricing, room mix, neighborhood performance, and review signals in one focused dashboard with a clean login flow.",
        "vi": "Khám phá giá, cơ cấu phòng, hiệu quả khu vực và tín hiệu đánh giá trong một dashboard tập trung với luồng đăng nhập gọn gàng.",
    },
    "login.chip.charts": {
        "en": "Plotly charts",
        "vi": "Biểu đồ Plotly",
    },
    "login.chip.preprocessing": {
        "en": "Preprocessing summary",
        "vi": "Tóm tắt tiền xử lý",
    },
    "login.chip.chatbot": {
        "en": "Insight chatbot",
        "vi": "Chatbot giải thích insight",
    },
    "login.section": {
        "en": "Inside the dashboard",
        "vi": "Bên trong dashboard",
    },
    "login.card.price.eyebrow": {
        "en": "Price",
        "vi": "Giá",
    },
    "login.card.price.title": {
        "en": "Revenue patterns",
        "vi": "Mẫu hình doanh thu",
    },
    "login.card.price.body": {
        "en": "Read distribution, outliers, and value ranges across the Airbnb listings.",
        "vi": "Xem phân phối, ngoại lệ và dải giá trị trên toàn bộ listing Airbnb.",
    },
    "login.card.room.eyebrow": {
        "en": "Room mix",
        "vi": "Cơ cấu phòng",
    },
    "login.card.room.title": {
        "en": "Listing balance",
        "vi": "Cân bằng danh mục",
    },
    "login.card.room.body": {
        "en": "Compare entire homes, private rooms, and shared rooms side by side.",
        "vi": "So sánh nhà nguyên căn, phòng riêng và phòng dùng chung trên cùng một màn hình.",
    },
    "login.card.demand.eyebrow": {
        "en": "Demand",
        "vi": "Nhu cầu",
    },
    "login.card.demand.title": {
        "en": "Review signals",
        "vi": "Tín hiệu đánh giá",
    },
    "login.card.demand.body": {
        "en": "Use reviews and availability to support market conclusions.",
        "vi": "Dùng đánh giá và mức sẵn có để củng cố kết luận thị trường.",
    },
    "login.note.title": {
        "en": "Welcome back",
        "vi": "Chào mừng quay lại",
    },
    "login.note.body": {
        "en": "Use your local account to access the full dashboard.",
        "vi": "Dùng tài khoản cục bộ của bạn để truy cập toàn bộ dashboard.",
    },
    "login.username": {
        "en": "Username",
        "vi": "Tên đăng nhập",
    },
    "login.password": {
        "en": "Password",
        "vi": "Mật khẩu",
    },
    "login.submit": {
        "en": "Login",
        "vi": "Đăng nhập",
    },
    "login.switch.title": {
        "en": "Need a new account?",
        "vi": "Cần tạo tài khoản mới?",
    },
    "login.switch.body": {
        "en": "Create one in the register page, then return here to sign in.",
        "vi": "Hãy tạo tài khoản ở trang đăng ký rồi quay lại đây để đăng nhập.",
    },
    "login.switch.button": {
        "en": "Open register page",
        "vi": "Mở trang đăng ký",
    },
    "register.badge": {
        "en": "Create your workspace access",
        "vi": "Tạo quyền truy cập không gian làm việc",
    },
    "register.title": {
        "en": "Set up an account before entering the dashboard.",
        "vi": "Thiết lập tài khoản trước khi vào dashboard.",
    },
    "register.body": {
        "en": "Register once to unlock the full Airbnb analytics flow, from raw data inspection to preprocessing, EDA, conclusions, and chatbot support.",
        "vi": "Đăng ký một lần để mở toàn bộ luồng phân tích Airbnb, từ xem dữ liệu gốc đến tiền xử lý, EDA, kết luận và chatbot hỗ trợ.",
    },
    "register.chip.local": {
        "en": "One local account",
        "vi": "Một tài khoản cục bộ",
    },
    "register.chip.session": {
        "en": "Session-based access",
        "vi": "Truy cập theo phiên",
    },
    "register.chip.return": {
        "en": "Fast return to analysis",
        "vi": "Quay lại phân tích nhanh",
    },
    "register.section": {
        "en": "How access works",
        "vi": "Cách quyền truy cập hoạt động",
    },
    "register.card.create.title": {
        "en": "Create credentials",
        "vi": "Tạo thông tin đăng nhập",
    },
    "register.card.create.body": {
        "en": "Choose a username and password that you can reuse whenever you return.",
        "vi": "Chọn tên đăng nhập và mật khẩu để có thể dùng lại mỗi khi quay lại.",
    },
    "register.card.open.title": {
        "en": "Open the dashboard",
        "vi": "Mở dashboard",
    },
    "register.card.open.body": {
        "en": "After registering, the app sends you back to login so you can enter immediately.",
        "vi": "Sau khi đăng ký, ứng dụng sẽ đưa bạn về trang đăng nhập để vào ngay lập tức.",
    },
    "register.card.explore.title": {
        "en": "Start exploring",
        "vi": "Bắt đầu khám phá",
    },
    "register.card.explore.body": {
        "en": "Move from overview to EDA and conclusion pages without exposing the data publicly.",
        "vi": "Di chuyển từ tổng quan đến EDA và kết luận mà không làm lộ dữ liệu ra ngoài.",
    },
    "register.note.title": {
        "en": "Create your account",
        "vi": "Tạo tài khoản của bạn",
    },
    "register.note.body": {
        "en": "Register a local user for this Streamlit dashboard.",
        "vi": "Đăng ký người dùng cục bộ cho dashboard Streamlit này.",
    },
    "register.rule.username": {
        "en": "Username must be at least 3 characters.",
        "vi": "Tên đăng nhập phải có ít nhất 3 ký tự.",
    },
    "register.rule.password": {
        "en": "Password must be at least 6 characters.",
        "vi": "Mật khẩu phải có ít nhất 6 ký tự.",
    },
    "register.rule.confirm": {
        "en": "Confirm password must match exactly.",
        "vi": "Mật khẩu xác nhận phải khớp hoàn toàn.",
    },
    "register.confirm_password": {
        "en": "Confirm Password",
        "vi": "Xác nhận mật khẩu",
    },
    "register.submit": {
        "en": "Create account",
        "vi": "Tạo tài khoản",
    },
    "register.switch.title": {
        "en": "Already registered?",
        "vi": "Đã có tài khoản?",
    },
    "register.switch.body": {
        "en": "Go back to login and enter the dashboard with your existing account.",
        "vi": "Quay lại đăng nhập và vào dashboard bằng tài khoản hiện có.",
    },
    "register.switch.button": {
        "en": "Open login page",
        "vi": "Mở trang đăng nhập",
    },
    "overview.hero_body": {
        "en": "Track price levels, room mix, neighborhood performance, and demand signals from a single dashboard. Current source: <strong>{source}</strong>.",
        "vi": "Theo dõi mức giá, cơ cấu phòng, hiệu quả khu vực và tín hiệu nhu cầu trong một dashboard duy nhất. Nguồn hiện tại: <strong>{source}</strong>.",
    },
    "overview.sample_info": {
        "en": "`data/Airbnb_Open_Data.csv` was not found, so the app is using sample data for the scaffold.",
        "vi": "Không tìm thấy `data/Airbnb_Open_Data.csv`, nên ứng dụng đang dùng dữ liệu mẫu cho bộ khung hiện tại.",
    },
    "overview.metric.listings": {
        "en": "Listings",
        "vi": "Listing",
    },
    "overview.metric.median_price": {
        "en": "Median Price",
        "vi": "Giá trung vị",
    },
    "overview.metric.avg_reviews": {
        "en": "Avg Reviews",
        "vi": "Đánh giá trung bình",
    },
    "overview.metric.avg_availability": {
        "en": "Avg Availability",
        "vi": "Sẵn có trung bình",
    },
    "overview.chart.price_distribution": {
        "en": "Nightly price distribution",
        "vi": "Phân phối giá theo đêm",
    },
    "overview.chart.room_mix": {
        "en": "Room type mix",
        "vi": "Cơ cấu loại phòng",
    },
    "overview.key_insights": {
        "en": "Key insights",
        "vi": "Insight chính",
    },
    "overview.neighborhood_snapshot": {
        "en": "Neighborhood price snapshot",
        "vi": "Ảnh chụp nhanh giá theo khu vực",
    },
    "raw.title": {
        "en": "Raw Data Explorer",
        "vi": "Khám phá dữ liệu gốc",
    },
    "raw.caption": {
        "en": "Use the filters to inspect the cleaned analytical dataset while still comparing it to the raw source.",
        "vi": "Dùng bộ lọc để xem bộ dữ liệu đã làm sạch đồng thời vẫn so sánh với nguồn gốc ban đầu.",
    },
    "raw.tab.upload": {
        "en": "Upload CSV",
        "vi": "Tải CSV",
    },
    "raw.tab.metrics": {
        "en": "Row + Column Count",
        "vi": "Số dòng + Số cột",
    },
    "raw.tab.overview": {
        "en": "Overview",
        "vi": "Tổng quan",
    },
    "raw.tab.preview": {
        "en": "Raw Preview",
        "vi": "Xem dữ liệu gốc",
    },
    "raw.tab.missing": {
        "en": "Missing Values",
        "vi": "Giá trị thiếu",
    },
    "raw.tab.pipeline": {
        "en": "Preprocessing",
        "vi": "Tiền xử lý",
    },
    "raw.tab.cleaned": {
        "en": "Cleaned Data",
        "vi": "Dữ liệu đã làm sạch",
    },
    "raw.upload.label": {
        "en": "Upload CSV for raw data audit",
        "vi": "Tải CSV để kiểm tra dữ liệu gốc",
    },
    "raw.upload.help": {
        "en": "Upload a CSV to inspect its row count, column count, and missing values by column.",
        "vi": "Tải một tệp CSV để xem số dòng, số cột và số giá trị thiếu theo từng cột.",
    },
    "raw.upload.error": {
        "en": "Unable to read the uploaded CSV: {error}",
        "vi": "Không thể đọc tệp CSV đã tải lên: {error}",
    },
    "raw.audit.title": {
        "en": "Raw data audit",
        "vi": "Kiểm tra dữ liệu gốc",
    },
    "raw.source.default": {
        "en": "Showing the raw dataset currently loaded in the app.",
        "vi": "Đang hiển thị bộ dữ liệu gốc hiện được nạp trong ứng dụng.",
    },
    "raw.source.uploaded": {
        "en": "Showing uploaded file: {file_name}",
        "vi": "Đang hiển thị tệp đã tải lên: {file_name}",
    },
    "raw.metric.rows": {
        "en": "Row Count",
        "vi": "Số dòng",
    },
    "raw.metric.columns": {
        "en": "Column Count",
        "vi": "Số cột",
    },
    "raw.preview.title": {
        "en": "Raw dataset preview",
        "vi": "Xem trước dữ liệu gốc",
    },
    "raw.missing.title": {
        "en": "Missing values by column",
        "vi": "Giá trị thiếu theo cột",
    },
    "raw.overview.title": {
        "en": "Overview data",
        "vi": "Tổng quan dữ liệu",
    },
    "raw.overview.caption": {
        "en": "Column reference for the full Airbnb dataset based on the preprocessing specification.",
        "vi": "Bảng tham chiếu cột cho toàn bộ bộ dữ liệu Airbnb dựa trên đặc tả tiền xử lý.",
    },
    "raw.pipeline.title": {
        "en": "Preprocessing pipeline",
        "vi": "Pipeline tiền xử lý",
    },
    "raw.pipeline.caption": {
        "en": "Review the 15 preprocessing steps and the matching code snippets from the project specification.",
        "vi": "Xem lại 15 bước tiền xử lý và đoạn mã tương ứng từ đặc tả của dự án.",
    },
    "raw.cleaned.title": {
        "en": "Cleaned dataset explorer",
        "vi": "Khám phá dữ liệu đã làm sạch",
    },
    "raw.filter.neighborhood_group": {
        "en": "Neighborhood Group",
        "vi": "Nhóm khu vực",
    },
    "raw.filter.room_type": {
        "en": "Room Type",
        "vi": "Loại phòng",
    },
    "raw.filter.price_range": {
        "en": "Price Range",
        "vi": "Khoảng giá",
    },
    "raw.filter.minimum_reviews": {
        "en": "Minimum Reviews",
        "vi": "Số đánh giá tối thiểu",
    },
    "raw.metric.filtered_rows": {
        "en": "Filtered Rows",
        "vi": "Dòng sau lọc",
    },
    "raw.download": {
        "en": "Download filtered data",
        "vi": "Tải dữ liệu đã lọc",
    },
    "raw.download_filename": {
        "en": "filtered_airbnb_data.csv",
        "vi": "du_lieu_airbnb_da_loc.csv",
    },
    "raw.expander": {
        "en": "Preview raw imported data",
        "vi": "Xem trước dữ liệu nhập gốc",
    },
    "prep.title": {
        "en": "Preprocessing",
        "vi": "Tiền xử lý",
    },
    "prep.caption": {
        "en": "This page documents how the raw Airbnb data is normalized into an analysis-ready table.",
        "vi": "Trang này mô tả cách dữ liệu Airbnb thô được chuẩn hóa thành bảng sẵn sàng cho phân tích.",
    },
    "prep.metric.rows_before": {
        "en": "Rows Before",
        "vi": "Dòng trước xử lý",
    },
    "prep.metric.rows_after": {
        "en": "Rows After",
        "vi": "Dòng sau xử lý",
    },
    "prep.metric.duplicates_removed": {
        "en": "Duplicates Removed",
        "vi": "Bản ghi trùng đã xóa",
    },
    "prep.metric.invalid_prices_removed": {
        "en": "Invalid Prices Removed",
        "vi": "Giá không hợp lệ đã xóa",
    },
    "prep.workflow_title": {
        "en": "Cleaning workflow",
        "vi": "Quy trình làm sạch",
    },
    "prep.workflow_body": {
        "en": "Column names are normalized to snake_case, currency values are converted to numerics, text gaps in location and room fields are filled with \"Unknown\", and rows with missing or non-positive prices are removed from the analytical dataset.",
        "vi": "Tên cột được chuẩn hóa sang snake_case, giá trị tiền tệ được chuyển thành số, chỗ trống trong cột vị trí và loại phòng được điền bằng \"Không xác định\", và các dòng có giá thiếu hoặc không dương sẽ bị loại khỏi bộ dữ liệu phân tích.",
    },
    "prep.missing_before": {
        "en": "Missing values before cleaning",
        "vi": "Giá trị thiếu trước làm sạch",
    },
    "prep.missing_after": {
        "en": "Missing values after cleaning",
        "vi": "Giá trị thiếu sau làm sạch",
    },
    "prep.preview": {
        "en": "Cleaned dataset preview",
        "vi": "Xem trước dữ liệu đã làm sạch",
    },
    "prep.schema": {
        "en": "Prepared schema",
        "vi": "Schema sau chuẩn bị",
    },
    "eda.title": {
        "en": "EDA Insights",
        "vi": "Phân tích EDA",
    },
    "eda.caption": {
        "en": "Interactive charts focus on pricing, room mix, neighborhood performance, and demand signals.",
        "vi": "Các biểu đồ tương tác tập trung vào giá, cơ cấu phòng, hiệu quả khu vực và tín hiệu nhu cầu.",
    },
    "eda.chart.price_by_room": {
        "en": "Price spread by room type",
        "vi": "Phân tán giá theo loại phòng",
    },
    "eda.chart.median_price_by_area": {
        "en": "Median price by neighborhood group",
        "vi": "Giá trung vị theo nhóm khu vực",
    },
    "eda.chart.reviews_vs_price": {
        "en": "Reviews vs nightly price",
        "vi": "Đánh giá so với giá theo đêm",
    },
    "eda.chart.availability_by_room": {
        "en": "Availability by room type",
        "vi": "Mức sẵn có theo loại phòng",
    },
    "conclusion.title": {
        "en": "Conclusion",
        "vi": "Kết luận",
    },
    "conclusion.caption": {
        "en": "Summarized findings help frame the dataset into practical takeaways.",
        "vi": "Các phát hiện được tóm tắt giúp chuyển bộ dữ liệu thành những điểm rút ra thực tế.",
    },
    "conclusion.insight_label": {
        "en": "Insight {index}",
        "vi": "Insight {index}",
    },
    "conclusion.next_steps_title": {
        "en": "Recommended next steps",
        "vi": "Bước tiếp theo được khuyến nghị",
    },
    "conclusion.next_steps_body": {
        "en": "Add the full Airbnb CSV, extend the preprocessing logic for more columns, and connect the chatbot to a real LLM if you want natural-language explanations beyond this scaffold.",
        "vi": "Hãy thêm file CSV Airbnb đầy đủ, mở rộng logic tiền xử lý cho nhiều cột hơn và kết nối chatbot với một LLM thật nếu bạn muốn phần giải thích ngôn ngữ tự nhiên vượt ngoài bộ khung này.",
    },
    "chatbot.title": {
        "en": "Chatbot",
        "vi": "Trợ lý AI",
    },
    "chatbot.caption": {
        "en": "This scaffold uses a rule-based assistant that answers questions using the current dataset summary.",
        "vi": "Bộ khung này dùng trợ lý dựa trên luật để trả lời câu hỏi bằng phần tóm tắt của bộ dữ liệu hiện tại.",
    },
    "chatbot.quick.price": {
        "en": "What does the price distribution look like?",
        "vi": "Phân phối giá trông như thế nào?",
    },
    "chatbot.quick.room_mix": {
        "en": "Which room type is most common?",
        "vi": "Loại phòng nào phổ biến nhất?",
    },
    "chatbot.quick.expensive_area": {
        "en": "Which neighborhood group is the most expensive?",
        "vi": "Nhóm khu vực nào đắt nhất?",
    },
    "chatbot.quick.availability": {
        "en": "How available are listings across the year?",
        "vi": "Listing còn trống trong năm ở mức nào?",
    },
    "chatbot.input": {
        "en": "Ask about prices, room types, reviews, neighborhoods, or availability.",
        "vi": "Hãy hỏi về giá, loại phòng, đánh giá, khu vực hoặc mức sẵn có.",
    },
    "auth.error.username_short": {
        "en": "Username must be at least 3 characters long.",
        "vi": "Tên đăng nhập phải có ít nhất 3 ký tự.",
    },
    "auth.error.username_spaces": {
        "en": "Username cannot contain spaces.",
        "vi": "Tên đăng nhập không được chứa khoảng trắng.",
    },
    "auth.error.password_short": {
        "en": "Password must be at least 6 characters long.",
        "vi": "Mật khẩu phải có ít nhất 6 ký tự.",
    },
    "auth.error.password_mismatch": {
        "en": "Passwords do not match.",
        "vi": "Mật khẩu không khớp.",
    },
    "auth.error.username_exists": {
        "en": "That username already exists.",
        "vi": "Tên đăng nhập đó đã tồn tại.",
    },
    "auth.notice.account_created": {
        "en": "Account created. You can now log in.",
        "vi": "Tài khoản đã được tạo. Bây giờ bạn có thể đăng nhập.",
    },
    "auth.error.invalid_login": {
        "en": "Invalid username or password.",
        "vi": "Tên đăng nhập hoặc mật khẩu không hợp lệ.",
    },
    "auth.notice.welcome_back": {
        "en": "Welcome back, {username}.",
        "vi": "Chào mừng quay lại, {username}.",
    },
    "insight.no_data": {
        "en": "No data is available yet. Add `data/Airbnb_Open_Data.csv` to unlock the full dashboard.",
        "vi": "Hiện chưa có dữ liệu. Hãy thêm `data/Airbnb_Open_Data.csv` để mở toàn bộ dashboard.",
    },
    "insight.typical_price": {
        "en": "The typical nightly price is {median}, with an average of {mean}.",
        "vi": "Mức giá theo đêm điển hình là {median}, với giá trung bình là {mean}.",
    },
    "insight.top_area": {
        "en": "{area} has the highest median price at {price}, making it the premium market in this dataset.",
        "vi": "{area} có giá trung vị cao nhất ở mức {price}, trở thành thị trường cao cấp nhất trong bộ dữ liệu này.",
    },
    "insight.room_mix": {
        "en": "{room_type} is the dominant room type, accounting for {share}% of active listings.",
        "vi": "{room_type} là loại phòng chiếm ưu thế, chiếm {share}% số listing đang hoạt động.",
    },
    "insight.review_rank": {
        "en": "{room_type} has the strongest median review volume at {reviews} reviews, which is a useful proxy for guest demand.",
        "vi": "{room_type} có lượng đánh giá trung vị cao nhất ở mức {reviews} đánh giá, là một chỉ báo hữu ích cho nhu cầu của khách.",
    },
    "insight.availability_correlation": {
        "en": "Availability and price show a {strength} {direction} relationship ({correlation}), so supply dynamics are worth monitoring.",
        "vi": "Mức sẵn có và giá cho thấy mối quan hệ {direction} ở mức {strength} ({correlation}), vì vậy động lực cung là yếu tố đáng theo dõi.",
    },
    "insight.strength.weak": {
        "en": "weak",
        "vi": "yếu",
    },
    "insight.strength.moderate": {
        "en": "moderate",
        "vi": "vừa",
    },
    "insight.strength.strong": {
        "en": "strong",
        "vi": "mạnh",
    },
    "insight.direction.positive": {
        "en": "positive",
        "vi": "thuận chiều",
    },
    "insight.direction.negative": {
        "en": "negative",
        "vi": "nghịch chiều",
    },
    "chat.answer.price": {
        "en": "The median price is {median} and the average is {mean}. The 75th percentile is {percentile}.",
        "vi": "Giá trung vị là {median} và giá trung bình là {mean}. Phân vị thứ 75 là {percentile}.",
    },
    "chat.answer.room_mix": {
        "en": "Room type mix: {summary}.",
        "vi": "Cơ cấu loại phòng: {summary}.",
    },
    "chat.answer.top_areas": {
        "en": "The highest-priced areas by median price are {areas}.",
        "vi": "Các khu vực có giá trung vị cao nhất là {areas}.",
    },
    "chat.answer.reviews": {
        "en": "The typical listing has {median} reviews. The top decile starts around {top_decile} reviews.",
        "vi": "Listing điển hình có {median} đánh giá. Nhóm 10% cao nhất bắt đầu từ khoảng {top_decile} đánh giá.",
    },
    "chat.answer.availability": {
        "en": "Median availability is {median} days per year, while the average is {mean} days.",
        "vi": "Mức sẵn có trung vị là {median} ngày mỗi năm, trong khi trung bình là {mean} ngày.",
    },
}

ROOM_TYPE_TRANSLATIONS = {
    "Entire home/apt": {
        "en": "Entire home/apt",
        "vi": "Nhà/căn hộ nguyên căn",
    },
    "Private room": {
        "en": "Private room",
        "vi": "Phòng riêng",
    },
    "Shared room": {
        "en": "Shared room",
        "vi": "Phòng dùng chung",
    },
    "Hotel room": {
        "en": "Hotel room",
        "vi": "Phòng khách sạn",
    },
    "Unknown": {
        "en": "Unknown",
        "vi": "Không xác định",
    },
}

COLUMN_TRANSLATIONS = {
    "id": {"en": "ID", "vi": "ID"},
    "name": {"en": "Name", "vi": "Tên"},
    "host_name": {"en": "Host Name", "vi": "Tên chủ nhà"},
    "neighbourhood_group": {"en": "Neighborhood Group", "vi": "Nhóm khu vực"},
    "neighbourhood": {"en": "Neighborhood", "vi": "Khu vực"},
    "room_type": {"en": "Room Type", "vi": "Loại phòng"},
    "price": {"en": "Price", "vi": "Giá"},
    "number_of_reviews": {"en": "Number of Reviews", "vi": "Số đánh giá"},
    "reviews_per_month": {"en": "Reviews per Month", "vi": "Đánh giá mỗi tháng"},
    "review_rate_number": {"en": "Review Score", "vi": "Điểm đánh giá"},
    "minimum_nights": {"en": "Minimum Nights", "vi": "Số đêm tối thiểu"},
    "availability_365": {"en": "Availability 365", "vi": "Số ngày còn trống/năm"},
    "last_review": {"en": "Last Review", "vi": "Đánh giá gần nhất"},
    "column": {"en": "Column", "vi": "Cột"},
    "original_dtype": {"en": "Original Data Type", "vi": "Kiểu dữ liệu gốc"},
    "meaning": {"en": "Meaning", "vi": "Ý nghĩa"},
    "handling": {"en": "How to Handle", "vi": "Cách xử lý"},
    "missing_values": {"en": "Missing Values", "vi": "Giá trị thiếu"},
    "missing_pct": {"en": "Missing %", "vi": "Tỷ lệ thiếu %"},
    "median": {"en": "Median", "vi": "Trung vị"},
    "mean": {"en": "Mean", "vi": "Trung bình"},
    "count": {"en": "Count", "vi": "Số lượng"},
    "dtype": {"en": "Data Type", "vi": "Kiểu dữ liệu"},
}


def get_language() -> str:
    return st.session_state.get("language", DEFAULT_LANGUAGE)


def t(key: str, language: str | None = None, **kwargs: object) -> str:
    lang = language or get_language()
    entry = TRANSLATIONS.get(key, {})
    template = entry.get(lang) or entry.get(DEFAULT_LANGUAGE) or key
    return template.format(**kwargs)


def language_name(code: str) -> str:
    return LANGUAGE_OPTIONS.get(code, code)


def render_language_selector(
    *,
    key: str,
    compact: bool = False,
) -> None:
    current = get_language()
    next_language = "vi" if current == "en" else "en"
    selected = st.button(
        "\u00a0" if compact else t("common.language_toggle"),
        key=key,
        icon=":material/language:",
        type="tertiary",
        help=t(
            "common.switch_language_help",
            current=language_name(current),
            next=language_name(next_language),
        ),
        width="content",
    )
    if selected:
        st.session_state["language"] = next_language
        st.session_state["chat_history"] = []
        st.rerun()


def get_app_title(language: str | None = None) -> str:
    return t("app.title", language=language)


def nav_label(page_key: str, language: str | None = None) -> str:
    return t(f"nav.{page_key}.label", language=language)


def display_source_label(source_label: str, language: str | None = None) -> str:
    if source_label == SAMPLE_SOURCE_LABEL:
        return t("source.sample", language=language)
    return source_label


def translate_room_type(value: object, language: str | None = None) -> str:
    text = str(value)
    lang = language or get_language()
    return ROOM_TYPE_TRANSLATIONS.get(text, {}).get(lang, text)


def localize_dataframe_for_display(frame: pd.DataFrame, language: str | None = None) -> pd.DataFrame:
    lang = language or get_language()
    localized = frame.copy()
    if "room_type" in localized.columns:
        localized["room_type"] = localized["room_type"].map(lambda value: translate_room_type(value, lang))

    rename_map = {
        column: COLUMN_TRANSLATIONS.get(column, {}).get(lang, column)
        for column in localized.columns
    }
    return localized.rename(columns=rename_map)
