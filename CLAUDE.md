# Airbnb Data Dashboard — Project Spec

## Project Overview

Streamlit dashboard phân tích dữ liệu Airbnb NYC. Gồm 3 tab chính: Raw Data → Process Data → EDA Data.

## File Structure

```
data/
├── Airbnb_Open_Data.csv          # Raw input
├── Airbnb_Data_processed.csv     # Sau processing (chưa encoding)
└── Airbnb_Data_cleaned.csv       # Sau encoding (dùng cho EDA)
pages/
├── data_raw.py                   # Tab Raw Data
├── preprocessing.py              # Tab Process Data
└── eda.py                        # Tab EDA Data
```

## Flow

```
Raw Data Tab → Process Data Tab → EDA Data Tab
```

---

## TAB 1: Raw Data (`pages/data_raw.py`)

### Buttons
- **Process Data Button** → chuyển sang Process Data Tab

### I. Size of Data
- Số Record
- Số Column

### II. Health of Data
Bảng hiển thị theo chiều dọc (mỗi hàng = 1 cột), gồm:
- Column name
- Type of data
- Null count
- Percent of Null (%)
- Thống kê các cột Numerical (min, max, mean, median, std)

### III. Health of Data Visualization
- Biểu đồ Null các cột (bar chart ngang)
- Biểu đồ Boxplot các cột numerical

---

## TAB 2: Process Data (`pages/preprocessing.py`)

### A. Processing Pipeline (chạy tự động khi vào tab)

#### Bước 1 — Remove Columns
Xóa các cột:
- `country`
- `country code`
- `house_rules`
- `license`

#### Bước 2 — Remove Records (Null lat/long)
- Xóa 8 dòng có `lat` null
- Xóa 8 dòng có `long` null

#### Bước 3 — Data Cleaning (từng cột)

| # | Cột | Xử lý |
|---|-----|-------|
| 1 | id | Chuyển về string/object. Kiểm tra duplicate, bỏ dòng trùng |
| 2 | NAME | Strip whitespace, xóa emoji/ký tự đặc biệt, lowercase |
| 3 | host id | Chuyển về object |
| 4 | host_identity_verified | Fill null = "unconfirmed" |
| 5 | host name | Strip, fill null = "unknown", lowercase |
| 6 | neighbourhood group | Strip, fix typos (brookln→Brooklyn, manhatan→Manhattan), fill null = "Other" |
| 7 | neighbourhood | Fill null = "Other" |
| 8 | instant_bookable | Fill null = "FALSE" |
| 9 | cancellation_policy | Fill null = "unknown" |
| 10 | room type | Strip, Title Case |
| 11 | Construction year | Chuyển về int, fill null = median |
| 12 | price | Bỏ $ và dấu ",", chuyển float64, fill null = median |
| 13 | service fee | Bỏ $ và dấu ",", chuyển float64, fill null = median |
| 14 | minimum nights | Xóa giá trị > 365 hoặc < 0, fill null = median |
| 15 | number of reviews | Fill null = 0 |
| 16 | last review | Chuyển datetime, null để trống |
| 17 | reviews per month | Fill null = 0 |
| 18 | review rate number | Fill null = median |
| 19 | calculated host listings count | Fill null = 1 |
| 20 | availability 365 | Clamp về [0, 365], fill null = 0 |

#### Bước 4 — Feature Engineering
Tạo cột mới `occupancy_rate` (Tỷ lệ lấp đầy):
```python
df['occupancy_rate'] = ((365 - df['availability_365']) / 365) * 100
```

### B. Display

#### Buttons
- **EDA Data Button** → chuyển sang EDA Data Tab
- **Download Processed Data Button** → tải file đã xử lý (chưa encoding)

#### I. Size of Processed Data
- Số Record
- Số Column

#### II. Health of Processed Data
Bảng theo chiều dọc:
- Column name
- Type of data
- Null count
- Percent of Null (%)
- Thống kê Numerical columns

#### III. Health of Processed Data Visualization
- Biểu đồ so sánh Null trước và sau xử lý (grouped bar chart)
- Biểu đồ Boxplot các cột numerical trước và sau xử lý

---

## TAB 3: EDA Data (`pages/eda.py`)

### Buttons
- **Download Encoding Data Button** → tải file đã Encoding
- **Log Out Button** → trở về màn hình Login

### I. Data Visualization (các biểu đồ theo thứ tự)

1. **Heatmap tương quan** giữa các cột
   - Kết luận: Chọn `occupancy_rate` làm cột Target vì tương quan nhiều nhất

2. **Tỷ lệ lấp đầy theo neighbourhood group**
   - Kết luận: Brooklyn có nhu cầu thực tế cao nhất, cao hơn Manhattan

3. **Tỷ lệ lấp đầy theo instant_bookable**
   - Kết luận: Chênh lệch không đáng kể

4. **Tỷ lệ lấp đầy theo cancellation_policy**
   - Kết luận: Không phải yếu tố tiên quyết

5. **Tỷ lệ lấp đầy theo room type**
   - Kết luận: Ở dài hạn cao hơn ngắn hạn, khách công tác > khách du lịch

6. **Tỷ lệ lấp đầy theo Construction year**
   - Kết luận: Biến động nhẹ 60.3%–62.6% từ 2003–2022

7. **Tỷ lệ lấp đầy theo price**
   - Kết luận: Giá không ảnh hưởng quá lớn

8. **Tỷ lệ lấp đầy theo service fee**
   - Kết luận: Không có tác động đáng kể

9. **Tỷ lệ lấp đầy theo minimum nights**
   - Kết luận: Yêu cầu ở quá nhiều đêm → tỷ lệ lấp đầy thấp hơn

10. **Tỷ lệ lấp đầy theo number of reviews**
    - Kết luận: Số review tỷ lệ thuận với tỷ lệ lấp đầy

11. **Tỷ lệ lấp đầy theo review rate number**
    - Kết luận: Điểm đánh giá đơn lẻ không phải yếu tố mạnh nhất

12. **Tỷ lệ lấp đầy theo calculated host listings count**
    - Kết luận: Chủ nhà quy mô lớn → tỷ lệ phòng trống cao hơn

---

## Dependencies

```python
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
```

## General Rules
- Giữ nguyên layout Streamlit và authentication flow hiện tại
- Không restructure project
- Mỗi tab là 1 file riêng trong `pages/`
- Dùng Plotly cho charts (interactive hơn)
