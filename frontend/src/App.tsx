import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { useAuth } from "@/hooks/useAuth"
import LoginPage from "@/pages/LoginPage"
import ChatPage from "@/pages/ChatPage"

export default function App() {
  const auth = useAuth()
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/login" element={<LoginPage auth={auth} />} />
        <Route path="/" element={<ChatPage auth={auth} />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}
