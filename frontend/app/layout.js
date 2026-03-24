import "./globals.css";

export const metadata = {
  title: "E-Commerce AI Assistant",
  description: "ChatGPT-style frontend for Agentic E-Commerce Assistant",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
