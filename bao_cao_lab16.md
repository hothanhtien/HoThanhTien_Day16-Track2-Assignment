# Báo Cáo Lab 16 - Cloud AI Environment Setup

**Họ và tên: Hồ Thành Tiến** 

**MSSV:** ***2A202600868***

## Thông tin triển khai

- **Phương án:** CPU Instance + LightGBM (thay thế GPU)
- **Instance CPU Node:** t3.micro (2 vCPU, 1GB RAM)
- **Instance Bastion:** t3.micro
- **Region:** us-east-1 (N. Virginia)
- **Cold Start Time:** ~5 phút (từ terraform apply đến API response đầu tiên)

## Lý do dùng CPU thay GPU

Tài khoản AWS sử dụng Free Plan, chỉ cho phép các instance thuộc danh sách free-tier eligible. Không thể chạy instance GPU `g4dn.xlarge` (yêu cầu quota đặc biệt) cũng như instance CPU cao cấp `r5.2xlarge` theo đề bài. Sau khi kiểm tra danh sách free-tier eligible, đã chọn `t3.micro` để triển khai phương án dự phòng với LightGBM thay vì vLLM.

## Kết quả Benchmark (LightGBM trên t3.micro)

| Metric                           | Kết quả |
| -------------------------------- | --------- |
| Thời gian load data             | 0.14s     |
| Thời gian training              | 0.67s     |
| Best iteration                   | 11        |
| AUC-ROC                          | 0.6047    |
| Accuracy                         | 98.42%    |
| F1-Score                         | 0.0814    |
| Precision                        | 0.07      |
| Recall                           | 0.0972    |
| Inference latency (1 row)        | 0.056ms   |
| Inference throughput (1000 rows) | 0.85ms    |

## Nhận xét

- Training time 0.67s là rất nhanh nhờ LightGBM tối ưu cho CPU.
- AUC-ROC 0.6047 ở mức trung bình do dataset synthetic bị imbalanced nặng (tỷ lệ fraud chỉ 0.2%).
- Inference latency 0.056ms/row cho thấy LightGBM phù hợp cho real-time prediction.
- So với GPU (g4dn.xlarge): LightGBM trên CPU đủ mạnh cho bài toán tabular data, không cần GPU.

## Chi phí

- Tài khoản Free Plan: $0 (credits cover all costs).
- Ước tính nếu dùng r5.2xlarge: ~$0.57/giờ (EC2 + NAT Gateway + ALB).
