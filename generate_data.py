import pandas as pd
import os

def generate_sample_data():
    """
    Sinh dữ liệu mẫu (QA) về bóng đá toàn cầu.
    Đã được nâng cấp với rất nhiều dữ kiện mới nhất năm 2024 (Euro, Copa, Chuyển nhượng).
    """
    
    data = [
        # --- KẾT QUẢ & CÁC GIẢI ĐẤU LỚN ---
        {"Question": "Ai vô địch Euro 2024?", "Context_Answer": "**Tây Ban Nha** đã giành chức vô địch Euro 2024 sau khi đánh bại Anh với tỷ số 2-1 trong trận chung kết. Mikel Oyarzabal là người ghi bàn quyết định.", "Topic": "Result"},
        {"Question": "Tỷ số trận chung kết Euro 2024?", "Context_Answer": "Trận chung kết Euro 2024 kết thúc với tỷ số **2-1** nghiêng về Tây Ban Nha trước đội tuyển Anh. Các bàn thắng của Tây Ban Nha được ghi do công của Nico Williams và Mikel Oyarzabal.", "Topic": "Result"},
        {"Question": "Đội nào vô địch Copa America 2024?", "Context_Answer": "**Argentina** đã bảo vệ thành công ngôi vương Copa America 2024 sau khi thắng Colombia 1-0 nhờ bàn thắng ở hiệp phụ của Lautaro Martinez.", "Topic": "Result"},
        {"Question": "Messi có đá trận chung kết Copa America không?", "Context_Answer": "Có, Lionel Messi có đá chính nhưng không may gặp chấn thương mắt cá và phải rời sân giữa chừng trong nước mắt. Dù vậy, đồng đội của anh vẫn giành chiến thắng.", "Topic": "Result"},
        {"Question": "Ai vô địch Ngoại hạng Anh mùa giải 2023-2024?", "Context_Answer": "**Manchester City** đã lên ngôi vô địch Ngoại hạng Anh 23/24, lập kỷ lục vô địch 4 năm liên tiếp. Họ kết thúc với 91 điểm.", "Topic": "Standings"},
        {"Question": "Bayer Leverkusen có thành tích gì nổi bật mùa giải vừa rồi?", "Context_Answer": "Bayer Leverkusen dưới sự dẫn dắt của Xabi Alonso đã vô địch **Bundesliga với thành tích bất bại** (chưa từng thua trận nào ở giải quốc nội).", "Topic": "Result"},
        {"Question": "Đội nào vô địch C1 năm 2024?", "Context_Answer": "**Real Madrid** đã đoạt chức vô địch UEFA Champions League lần thứ 15 sau khi đánh bại Borussia Dortmund 2-0 tại Wembley.", "Topic": "Result"},
        {"Question": "Kết quả trận chung kết C2 (Europa League)?", "Context_Answer": "Atalanta đánh bại Bayer Leverkusen 3-0 để giành ngôi vô địch Europa League, nhờ cú hat-trick lịch sử của **Ademola Lookman**.", "Topic": "Result"},
        {"Question": "Arsenal đứng thứ mấy mùa trước?", "Context_Answer": "Arsenal giành ngôi Á quân Ngoại hạng Anh mùa giải 23/24 với 89 điểm, chỉ kém Man City đúng 2 điểm.", "Topic": "Standings"},
        {"Question": "Manchester United vô địch cúp gì gần đây?", "Context_Answer": "Manchester United đã đánh bại Man City 2-1 để lên ngôi vô địch **FA Cup 2023-2024**, qua đó giành luôn vé dự Europa League.", "Topic": "Result"},
        {"Question": "Đội bóng nào vô địch V-League 2023/2024?", "Context_Answer": "**Thép Xanh Nam Định** đã giành chức vô địch V-League mùa giải 2023/2024 cực kỳ thuyết phục.", "Topic": "Result"},
        
        # --- VUA PHÁ LƯỚI & CẦU THỦ XUẤT SẮC ---
        {"Question": "Vua phá lưới Ngoại hạng Anh 23/24 là ai?", "Context_Answer": "**Erling Haaland** giành giải Vua phá lưới (Golden Boot) Premier League với 27 bàn thắng.", "Topic": "Scorers"},
        {"Question": "Ai là vua phá lưới Euro 2024?", "Context_Answer": "Có tới 6 cầu thủ cùng chia sẻ danh hiệu Vua phá lưới Euro 2024 với **3 bàn thắng**, bao gồm: \n* Cody Gakpo (Hà Lan)\n* Harry Kane (Anh)\n* Dani Olmo (Tây Ban Nha)\n* Jamal Musiala (Đức)\n* Ivan Schranz (Slovakia)\n* Georges Mikautadze (Georgia)", "Topic": "Scorers"},
        {"Question": "Cầu thủ xuất sắc nhất Euro 2024?", "Context_Answer": "**Rodri** (Tây Ban Nha) đã được UEFA bầu chọn là Cầu thủ xuất sắc nhất giải đấu (Player of the Tournament) tại Euro 2024.", "Topic": "Scorers"},
        {"Question": "Cầu thủ trẻ xuất sắc nhất Euro 2024?", "Context_Answer": "Thần đồng **Lamine Yamal** (Tây Ban Nha), người bước sang tuổi 17 ngay trước trận chung kết, đã giành giải Cầu thủ trẻ xuất sắc nhất.", "Topic": "Scorers"},
        {"Question": "Vua phá lưới Copa America 2024?", "Context_Answer": "**Lautaro Martinez** (Argentina) là vua phá lưới Copa America 2024 với 5 bàn thắng, bao gồm cả bàn quyết định trong trận chung kết.", "Topic": "Scorers"},
        {"Question": "Harry Kane ghi bao nhiêu bàn cho Bayern mùa đầu tiên?", "Context_Answer": "Harry Kane ghi **36 bàn** ở Bundesliga, đoạt luôn Chiếc giày vàng châu Âu mùa giải 2023/2024.", "Topic": "Scorers"},
        {"Question": "Vua phá lưới V-League 2023/2024?", "Context_Answer": "Tiền đạo **Rafaelson** (Nam Định) đã thiết lập kỷ lục ghi bàn khủng khiếp tại V-League với 31 bàn thắng.", "Topic": "Scorers"},
        
        # --- CHUYỂN NHƯỢNG (TRANSFERS HÈ 2024) ---
        {"Question": "Mbappe đã chuyển đến đội nào?", "Context_Answer": "Tin bom tấn: **Kylian Mbappe** đã chính thức ra mắt **Real Madrid** theo dạng chuyển nhượng tự do. Anh khoác chiếc áo số 9.", "Topic": "Transfers"},
        {"Question": "Arsenal mua ai hè này?", "Context_Answer": "Arsenal đã mua đứt thủ môn **David Raya** và chốt xong trung vệ **Riccardo Calafiori** (từ Bologna) sau kỳ Euro xuất sắc của anh.", "Topic": "Transfers"},
        {"Question": "Man Utd có tân binh nào mới?", "Context_Answer": "Manchester United đang hoạt động rất mạnh mẽ, họ đã chiêu mộ tiền đạo **Joshua Zirkzee** (Bologna) và tài năng trẻ người Pháp **Leny Yoro** (Lille).", "Topic": "Transfers"},
        {"Question": "Chelsea mua bán ra sao hè 2024?", "Context_Answer": "Chelsea tiếp tục chiến lược mua cầu thủ trẻ. Họ đã đón **Tosin Adarabioyo**, **Kiernan Dewsbury-Hall**, và **Marc Guiu**.", "Topic": "Transfers"},
        {"Question": "Douglas Luiz đi đâu?", "Context_Answer": "Tiền vệ trụ cột của Aston Villa, **Douglas Luiz**, đã chính thức chuyển sang khoác áo **Juventus** ở Serie A.", "Topic": "Transfers"},
        {"Question": "Palhinha đã tới Bayern chưa?", "Context_Answer": "Rồi, sau khi hụt vào phút chót mùa hè năm ngoái, mùa hè 2024 **Joao Palhinha** đã chính thức cập bến Bayern Munich từ Fulham.", "Topic": "Transfers"},
        {"Question": "Savinho chuyển đến đâu?", "Context_Answer": "Cầu thủ chạy cánh người Brazil, **Savinho** (tỏa sáng ở Girona), đã được **Manchester City** mua đứt để bổ sung cho hàng công.", "Topic": "Transfers"},
        {"Question": "Đội bóng nào mua Endrick?", "Context_Answer": "Thần đồng người Brazil **Endrick** đã chính thức hội quân và ra mắt **Real Madrid** sau khi đủ 18 tuổi.", "Topic": "Transfers"},
        {"Question": "Amadou Onana gia nhập đội bóng nào?", "Context_Answer": "**Aston Villa** đã chi 50 triệu bảng để chiêu mộ tiền vệ phòng ngự Amadou Onana từ Everton.", "Topic": "Transfers"},

        # --- HUẤN LUYỆN VIÊN (MANAGERS) ---
        {"Question": "Ai thay Jurgen Klopp?", "Context_Answer": "Sau khi Klopp chia tay, Liverpool đã bổ nhiệm HLV người Hà Lan **Arne Slot** (từ Feyenoord) lên dẫn dắt đội bóng.", "Topic": "Manager"},
        {"Question": "HLV mới của Chelsea là ai?", "Context_Answer": "Chelsea đã sa thải Pochettino và chọn **Enzo Maresca**, người vừa giúp Leicester City vô địch Championship, lên làm HLV trưởng.", "Topic": "Manager"},
        {"Question": "Bayern chọn ai làm HLV thay Tuchel?", "Context_Answer": "Thật bất ngờ, Bayern Munich đã quyết định bổ nhiệm cựu trung vệ **Vincent Kompany** làm tân HLV trưởng.", "Topic": "Manager"},
        {"Question": "Juventus có HLV mới không?", "Context_Answer": "Có, Juventus đã ký hợp đồng với **Thiago Motta** sau khi ông giúp Bologna có vé dự Champions League cực kỳ ấn tượng.", "Topic": "Manager"},
        {"Question": "ĐT Anh chia tay HLV nào?", "Context_Answer": "Sau thất bại ở chung kết Euro 2024, HLV **Gareth Southgate** đã chính thức từ chức thuyền trưởng ĐTQG Anh.", "Topic": "Manager"},
        {"Question": "HLV đội tuyển Việt Nam hiện tại là ai?", "Context_Answer": "HLV trưởng ĐTQG Việt Nam hiện tại là ông **Kim Sang-sik** người Hàn Quốc.", "Topic": "Manager"},
        
        # --- CHẤN THƯƠNG & KHÁC ---
        {"Question": "Tình hình chấn thương của Ronald Araujo?", "Context_Answer": "Hậu vệ Ronald Araujo (Barca) dính chấn thương gân kheo nặng khi đá cho Uruguay ở Copa America và phải phẫu thuật, dự kiến nghỉ 4 tháng.", "Topic": "Injuries"},
        {"Question": "Leny Yoro có bị chấn thương không?", "Context_Answer": "Không may cho MU, tân binh đắt giá **Leny Yoro** vừa gặp chấn thương rạn xương bàn chân trong trận giao hữu và có thể phải nghỉ tới 3 tháng.", "Topic": "Injuries"},
        {"Question": "Rasmus Hojlund bị gì?", "Context_Answer": "Hojlund cũng gặp chấn thương gân kheo trong đợt giao hữu hè của MU, dự kiến phải vắng mặt khoảng 6 tuần đầu mùa giải.", "Topic": "Injuries"},
        {"Question": "Luật việt vị mới có áp dụng không?", "Context_Answer": "FIFA và IFAB đang thử nghiệm luật việt vị mới của Arsene Wenger nhưng chưa chính thức áp dụng ở các giải đấu hàng đầu mùa 24/25. Hiện tại vẫn áp dụng công nghệ VAR bắt việt vị bán tự động (SAOT).", "Topic": "Cards"}
    ]

    df = pd.DataFrame(data)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/qa_train.csv', index=False, encoding='utf-8')
    print("Đã tạo file data/qa_train.csv thành công với", len(df), "bản ghi.")

if __name__ == "__main__":
    generate_sample_data()
