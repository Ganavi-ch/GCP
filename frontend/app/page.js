"use client";

import { useMemo, useState, useEffect, useRef } from "react";

export default function ChatPage() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content: "Hi! I am your E-Commerce AI Assistant. Ask me about coupons, orders, profile, or platform help."
    }
  ]);
  const [input, setInput] = useState("");
  const [userId, setUserId] = useState("u1001");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const messagesEndRef = useRef(null);

  const canSend = useMemo(() => input.trim().length > 0 && !loading, [input, loading]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  async function sendMessage(e) {
    if (e) e.preventDefault();
    const text = input.trim();
    if (!text || loading) return;

    setError("");
    setMessages((prev) => [...prev, { role: "user", content: text }]);
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          user_id: userId.trim() || null
        })
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status}`);
      }

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.answer }
      ]);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Connection failed";
      setError(msg);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: "I'm having trouble connecting to the store. Please ensure the backend is running." }
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="chat-root">
      <nav className="top-nav">
        <h1>AI Assistant</h1>
        <div className="user-settings">
          <span>User ID</span>
          <input
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="u1001"
          />
        </div>
      </nav>

      <section className="chat-panel">
        <div className="messages">
          {messages.map((msg, index) => (
            <article key={`${msg.role}-${index}`} className={`bubble ${msg.role}`}>
              <pre>{msg.content}</pre>
            </article>
          ))}
          {loading && (
            <div className="thinking bubble assistant">
              <div className="dot"></div>
              <div className="dot" style={{ animationDelay: '0.2s' }}></div>
              <div className="dot" style={{ animationDelay: '0.4s' }}></div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="composer-container">
          <form className="composer" onSubmit={sendMessage}>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask me anything in natural language..."
              rows={1}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault();
                  sendMessage();
                }
              }}
            />
            <button type="submit" disabled={!canSend}>
              Send
            </button>
          </form>
          {error && <div className="error-toast">{error}</div>}
        </div>
      </section>
    </main>
  );
}
