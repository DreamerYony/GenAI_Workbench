document.querySelector("form").addEventListener("submit", async (event) => {
    event.preventDefault();

    const userInput = document.querySelector("textarea[name='user_input']").value;
    const chatBox = document.querySelector(".chat-box");

    // 사용자의 메시지 추가
    const userMessage = document.createElement("div");
    userMessage.className = "chat-message user";
    userMessage.innerHTML = `
        <div class="message-bubble user">${userInput}</div>
        <div class="message-label user">사용자</div>
    `;
    chatBox.appendChild(userMessage);

    // 서버로 메시지 전송
    const response = await fetch("/ask", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ user_input: userInput }),
    });
    const html = await response.text();

    // 서버 응답에서 AI 메시지 추가
    const parser = new DOMParser();
    const doc = parser.parseFromString(html, "text/html");
    const aiMessage = doc.querySelector(".chat-box .chat-message.ai:last-child");
    chatBox.appendChild(aiMessage);

    // 스크롤 자동 이동
    chatBox.scrollTop = chatBox.scrollHeight;
    

    // 입력창 초기화
    document.querySelector("textarea[name='user_input']").value = "";
});