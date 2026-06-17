document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const typingIndicator = document.getElementById('typing-indicator');
    const clearChatBtn = document.getElementById('clear-chat-btn');

    // Mảng lưu lịch sử trò chuyện
    let chatHistory = [];

    // Cấu hình marked.js
    marked.setOptions({
        breaks: true, // Cho phép xuống dòng bằng \n
        gfm: true     // Hỗ trợ GitHub Flavored Markdown
    });

    // Hàm toàn cục để được gọi từ các thẻ li (Quick Suggestions)
    window.sendSuggestion = function(text) {
        userInput.value = text;
        sendMessage();
    };

    function appendMessage(sender, text, isHtml = false) {
        const messageDiv = document.createElement('div');
        messageDiv.classList.add('message');
        messageDiv.classList.add(sender === 'user' ? 'user-message' : 'bot-message');

        const avatarIcon = sender === 'user' ? 'fa-user' : 'fa-robot';
        
        let contentHtml = "";
        if (sender === 'user') {
            contentHtml = escapeHtml(text);
        } else {
            // Nếu là bot và có html content, thì render markdown
            contentHtml = isHtml ? text : marked.parse(text);
        }

        messageDiv.innerHTML = `
            <div class="avatar"><i class="fa-solid ${avatarIcon}"></i></div>
            <div class="message-content">${contentHtml}</div>
        `;
        
        chatBox.appendChild(messageDiv);
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }

    async function sendMessage() {
        const query = userInput.value.trim();
        if (!query) return;

        appendMessage('user', query);
        // Lưu vào lịch sử (tối đa 6 messages = 3 cặp QA)
        chatHistory.push({ role: 'user', content: query });
        if (chatHistory.length > 6) chatHistory.shift();

        userInput.value = '';
        typingIndicator.classList.remove('hidden');
        chatBox.scrollTop = chatBox.scrollHeight;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: query, history: chatHistory.slice(0, -1) }) // Không gửi lại query hiện tại trong history
            });

            const data = await response.json();
            typingIndicator.classList.add('hidden');

            if (data.error) {
                appendMessage('bot', data.error);
            } else {
                // Parse markdown từ answer của bot
                let htmlResponse = marked.parse(data.answer);
                
                // Thêm topic để demo kỹ thuật
                if (data.detected_topic && data.detected_topic !== "Error") {
                    htmlResponse += `<div class="context-info">🔍 Phân loại Intent: <span>${data.detected_topic}</span></div>`;
                }

                // Append html trực tiếp do marked.parse đã xử lý an toàn mức cơ bản (hoặc có thể dùng DOMPurify nếu cần)
                appendMessage('bot', htmlResponse, true);
                
                // Lưu vào lịch sử
                chatHistory.push({ role: 'model', content: data.answer });
                if (chatHistory.length > 6) chatHistory.shift();
            }
        } catch (error) {
            typingIndicator.classList.add('hidden');
            appendMessage('bot', 'Xin lỗi, không thể kết nối tới server. Vui lòng kiểm tra lại kết nối hoặc server đang tắt.');
            console.error('Error:', error);
        }
    }

    // Xử lý sự kiện nút Gửi
    sendBtn.addEventListener('click', sendMessage);

    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Xử lý sự kiện Clear Chat
    clearChatBtn.addEventListener('click', () => {
        // Chỉ xóa các thẻ .message, giữ lại các phần tử khác nếu có (như typing indicator)
        const messages = chatBox.querySelectorAll('.message');
        messages.forEach(msg => {
            // Bỏ qua tin nhắn chào mừng đầu tiên nếu muốn
            if (msg.classList.contains('bot-message') && msg === messages[0]) {
                return;
            }
            msg.remove();
        });
        chatHistory = []; // Xóa history
    });
});
